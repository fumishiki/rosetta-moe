use jlrs::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// --- Helper Extensions ---
trait JlrsResultExt<T> {
    fn into_result(self) -> Result<T>;
}

impl<T> JlrsResultExt<T> for JlrsResult<T> {
    fn into_result(self) -> Result<T> {
        self.map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

// --- Julia FFI Helpers ---
fn eval_julia<'scope>(frame: &mut GcFrame<'scope>, code: &str) -> Result<Value<'scope, 'static>> {
    // SAFETY: eval_string is called within a valid GcFrame scope
    unsafe { Value::eval_string(frame, code).into_result() }
}

fn get_mlcore<'scope>(frame: &GcFrame<'scope>) -> Result<Module<'scope>> {
    Module::main(frame)
        .submodule(frame, "MLCore")
        .map_err(|e| format!("MLCore not found: {}", e).into())
}

fn gpu_code(gpu: bool) -> &'static str {
    if gpu { "ps, st = (ps, st) .|> Lux.gpu" } else { "" }
}

// --- Config Types (mirroring Julia Config) ---
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab: u32,
    pub hidden: u32,
    pub ffn: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub experts: u32,
    pub topk: u32,
    pub loops: u32,
    pub ctx_len: u32,
    pub rope_base: f64,
    pub r: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: u32,
    pub learning_rate: f64,
    pub warmup_steps: u32,
    pub max_steps: u32,
    pub grad_clip: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchingConfig {
    pub enabled: bool,
    pub patcher_hidden: u32,
    pub threshold: f32,
    pub ema_momentum: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    pub models: std::collections::HashMap<String, ModelConfig>,
    pub training: TrainingConfig,
    pub patching: PatchingConfig,
}

impl FullConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    pub fn model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }
}

pub struct JuliaEngine {
    handle: LocalHandle,
    config: Option<FullConfig>,
    model_name: Option<String>,
}

impl JuliaEngine {
    pub fn new() -> Result<Self> {
        // SAFETY: Builder::start_local initializes Julia runtime in isolated thread-local state
        let handle = unsafe { Builder::new().start_local::<4>().into_result()? };
        Ok(Self { handle, config: None, model_name: None })
    }

    pub fn with_config(config: FullConfig, model_name: impl Into<String>) -> Result<Self> {
        let mut engine = Self::new()?;
        engine.config = Some(config);
        engine.model_name = Some(model_name.into());
        Ok(engine)
    }

    pub fn load_config(&mut self, path: impl AsRef<Path>, model_name: impl Into<String>) -> Result<&Self> {
        self.config = Some(FullConfig::load(path)?);
        self.model_name = Some(model_name.into());
        Ok(self)
    }

    pub fn load_module(&mut self, path: &str) -> Result<()> {
        self.handle.local_scope::<_, 2>(|mut frame| {
            eval_julia(&mut frame, &format!("include(\"{}\")", path))?;
            get_mlcore(&frame)?;
            Ok(())
        })
    }

    pub fn create_model(&mut self, use_patching: bool) -> Result<()> {
        let config = self.config.as_ref().ok_or("No config loaded. Call load_config first.")?;
        let model_name = self.model_name.as_ref().ok_or("No model name set.")?;

        self.handle.local_scope::<_, 4>(|mut frame| {
            // SAFETY: All FFI calls operate on valid GcFrame with proper lifetime management
            unsafe {
                let module = get_mlcore(&frame)?;
                let load_config_fn = module.global(&frame, "load_config")?.as_value();
                let model_name_str = JuliaString::new(&mut frame, model_name);
                let cfg = load_config_fn.call(&mut frame, &[model_name_str.as_value()]).into_result()?;

                let model_type = if use_patching { "BLTRecurrentMoEWithPatching" } else { "BLTRecurrentMoE" };
                let model_ctor = module.global(&frame, model_type)?.as_value();
                model_ctor.call(&mut frame, &[cfg]).into_result()?;

                eval_julia(&mut frame, &format!("global __rust_model = {}", model_type))?;
            }
            Ok(())
        })
    }

    pub fn forward(&mut self, byte_ids: &[u8], dims: [usize; 2], gpu: bool) -> Result<Vec<f32>> {
        self.handle.local_scope::<_, 4>(|mut frame| {
            // SAFETY: FFI calls with valid array slice and GcFrame scope
            unsafe {
                let ids_arr = TypedArray::from_slice(&mut frame, byte_ids, dims)?;
                let model = Module::main(&frame).global(&frame, "__rust_model")?.as_value();

                let setup_code = format!(
                    "rng = Random.default_rng(); ps, st = Lux.setup(rng, __rust_model); {}; (ps, st)",
                    gpu_code(gpu)
                );
                let (ps, st) = eval_julia(&mut frame, &setup_code)?.unbox::<(Value, Value)>()?;

                let (logits, _) = model.call(&mut frame, &[ids_arr.as_value(), ps, st])
                    .into_result()?
                    .unbox::<(Value, Value)>()?;

                let logits_cpu = if gpu { eval_julia(&mut frame, "logits |> Lux.cpu")? } else { logits };

                logits_cpu.cast::<TypedArray<f32>>()
                    .ok()
                    .and_then(|arr| arr.copy_inline_data().ok())
                    .map(|data| data.to_vec())
                    .ok_or_else(|| "Failed to convert logits to f32 array".into())
            }
        })
    }

    pub fn train_step(&mut self, ids: &[u8], targets: &[u8], dims: [usize; 2], lr: f64, gpu: bool) -> Result<f32> {
        self.handle.local_scope::<_, 6>(|mut frame| {
            // SAFETY: Array slices are valid for the scope, Julia code handles ownership
            unsafe {
                TypedArray::from_slice(&mut frame, ids, dims)?;
                TypedArray::from_slice(&mut frame, targets, dims)?;

                let train_code = format!(
                    "using Optimisers, Zygote; \
                     rng = Random.default_rng(); ps, st = Lux.setup(rng, __rust_model); {}; \
                     opt_st = Optimisers.setup(AdamW(eta={}), ps); \
                     loss_fn(ps) = (logits, _ = __rust_model(ids, ps, st); sum(logits)); \
                     grads = Zygote.gradient(loss_fn, ps)[1]; \
                     opt_st, ps = Optimisers.update(opt_st, ps, grads); \
                     loss_fn(ps)",
                    gpu_code(gpu), lr
                );

                eval_julia(&mut frame, &train_code)?.unbox::<f32>().map_err(Into::into)
            }
        })
    }

    pub fn generate(&mut self, prompt: &[u8], max_len: usize, temperature: f32, gpu: bool) -> Result<Vec<u8>> {
        self.handle.local_scope::<_, 4>(|mut frame| {
            // SAFETY: Prompt array is valid for scope, Julia handles generation loop
            unsafe {
                TypedArray::from_slice(&mut frame, prompt, [1, prompt.len()])?;

                let gen_code = format!(
                    "rng = Random.default_rng(); ps, st = Lux.setup(rng, __rust_model); {}; \
                     generate(prompt, max_len, temp) = (output = copy(prompt); \
                     for _ in 1:max_len; logits, _ = __rust_model(output, ps, st); \
                     probs = softmax(logits[:, end] ./ temp); next_token = rand(Categorical(probs)); \
                     output = hcat(output, next_token); end; output); \
                     generate(prompt, {}, {})",
                    gpu_code(gpu), max_len, temperature
                );

                eval_julia(&mut frame, &gen_code)?
                    .cast::<TypedArray<u8>>()
                    .ok()
                    .and_then(|arr| arr.copy_inline_data().ok())
                    .map(|data| data.to_vec())
                    .ok_or_else(|| "Failed to generate sequence".into())
            }
        })
    }

    pub fn model_config(&self) -> Option<&ModelConfig> {
        self.config
            .as_ref()
            .and_then(|cfg| self.model_name.as_ref().and_then(|name| cfg.model(name)))
    }

    pub fn training_config(&self) -> Option<&TrainingConfig> {
        self.config.as_ref().map(|cfg| &cfg.training)
    }

    pub fn patching_config(&self) -> Option<&PatchingConfig> {
        self.config.as_ref().map(|cfg| &cfg.patching)
    }

    pub fn with<F, R>(&mut self, f: F) -> Result<R>
    where
        F: for<'scope> FnOnce(&mut GcFrame<'scope>) -> Result<R>,
    {
        self.handle.local_scope::<_, 8>(|frame| f(frame))
    }
}

impl Default for JuliaEngine {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Julia runtime")
    }
}

// --- Builder Pattern ---
pub struct JuliaEngineBuilder {
    config_path: Option<String>,
    model_name: Option<String>,
    module_path: Option<String>,
    use_patching: bool,
}

impl JuliaEngineBuilder {
    pub const fn new() -> Self {
        Self { config_path: None, model_name: None, module_path: None, use_patching: false }
    }

    pub fn config(mut self, path: impl Into<String>) -> Self {
        self.config_path = Some(path.into());
        self
    }

    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    pub fn module(mut self, path: impl Into<String>) -> Self {
        self.module_path = Some(path.into());
        self
    }

    pub const fn with_patching(mut self, enabled: bool) -> Self {
        self.use_patching = enabled;
        self
    }

    pub fn build(self) -> Result<JuliaEngine> {
        let mut engine = JuliaEngine::new()?;

        if let Some(config_path) = self.config_path {
            let model_name = self.model_name.ok_or("Model name required when config is specified")?;
            engine.load_config(config_path, model_name)?;
        }

        if let Some(module_path) = self.module_path {
            engine.load_module(&module_path)?;
        }

        if engine.config.is_some() {
            engine.create_model(self.use_patching)?;
        }

        Ok(engine)
    }
}

impl Default for JuliaEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
