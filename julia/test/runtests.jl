# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

using Test
include(joinpath(@__DIR__, "..", "MoETransformer.jl"))
using .MoETransformer

@testset "MoE Transformer" begin
    @testset "Tensor" begin
        @testset "zeros" begin
            t = zeros_tensor(2, 3)
            @test numel(t) == 6
            @test all(t.data .== 0f0)
        end
        @testset "ones" begin
            t = ones_tensor(2, 3)
            @test all(t.data .== 1f0)
        end
        @testset "from_array" begin
            t = from_array(Float32[1 2 3; 4 5 6])
            @test size(t) == (2, 3)
            @test t.data[1] == 1f0
            @test t.data[6] == 6f0
        end
        @testset "add" begin
            a = from_array(Float32[1, 2, 3])
            b = from_array(Float32[4, 5, 6])
            c = a + b
            @test c.data == Float32[5, 7, 9]
        end
        @testset "scale" begin
            a = from_array(Float32[1, 2, 3])
            c = scale(a, 2f0)
            @test c.data == Float32[2, 4, 6]
        end
        @testset "silu" begin
            a = from_array(Float32[0, 1, -1])
            c = silu(a)
            @test abs(c.data[1]) < 0.001f0
            @test abs(c.data[2] - 0.731f0) < 0.01f0
        end
        @testset "softmax" begin
            a = from_array(reshape(Float32[1, 2, 3], 1, 3))
            c = softmax(a)
            @test abs(sum(c.data) - 1f0) < 0.001f0
            @test c.data[1] < c.data[2] < c.data[3]
        end
        @testset "matmul" begin
            a = from_array(reshape(Float32[1, 2, 3, 4, 5, 6], 2, 3))
            b = from_array(reshape(Float32[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 4))
            c = tensor_matmul(a, b)
            @test size(c) == (2, 4)
            @test c.data[1] == 22f0
        end
        @testset "transpose" begin
            a = from_array(reshape(Float32[1, 2, 3, 4, 5, 6], 2, 3))
            b = transpose_tensor(a)
            @test size(b) == (3, 2)
            @test b.data[1] == 1f0 && b.data[2] == 3f0 && b.data[3] == 5f0
        end
    end

    @testset "Layers" begin
        @testset "linear forward" begin
            input = from_array(reshape(rand(Float32, 2 * 4 * 8), 2, 4, 8))
            layer = Linear(8, 16, false)
            output = forward(layer, input)
            @test size(output) == (2, 4, 16)
        end
        @testset "rmsnorm shape" begin
            input = from_array(reshape(rand(Float32, 2 * 4 * 8), 2, 4, 8))
            norm = RMSNorm(8, 1f-6)
            output = forward(norm, input)
            @test size(output) == size(input)
        end
        @testset "swiglu shape" begin
            cfg = tiny()
            input = from_array(reshape(rand(Float32, 1 * 4 * cfg.hidden_dim), 1, 4, cfg.hidden_dim))
            ffn = SwiGLU(cfg.hidden_dim, cfg.ffn_dim)
            output = forward(ffn, input)
            @test size(output) == size(input)
        end
    end

    @testset "Model" begin
        @testset "config tiny" begin
            cfg = tiny()
            @test cfg.hidden_dim == 64
            @test cfg.n_layers == 2
            @test cfg.n_heads == 4
            @test cfg.vocab_size == 1000
        end
        @testset "model creation" begin
            m = tiny_model()
            @test m.config.hidden_dim == 64
        end
        @testset "forward shape" begin
            m = tiny_model()
            input = from_array(Float32[1 2 3 4])
            input = reshape_tensor(input, 1, 4)
            logits = forward(m, input)
            @test size(logits) == (1, 4, 1000)
        end
        @testset "generate length" begin
            m = tiny_model()
            tokens = generate_greedy(m, [1, 2, 3], 8)
            @test length(tokens) == 8
        end
        @testset "generate with GreedySampling" begin
            m = tiny_model()
            tokens = generate(m, [1, 2, 3], 8, GreedySampling())
            @test length(tokens) == 8
            tokens2 = generate_greedy(m, [1, 2, 3], 8)
            @test tokens == tokens2
        end
        @testset "block shape" begin
            cfg = tiny()
            block = TransformerBlock(cfg)
            input = from_array(reshape(randn(Float32, 1 * 4 * 64), 1, 4, 64))
            output = forward(block, input)
            @test size(output) == size(input)
            @test length(collect(parameters(block))) > 0
        end
        @testset "forward backward" begin
            m = tiny_model()
            input = from_array(Float32[10 20 30 40])
            input = reshape_tensor(input, 1, 4)
            logits = forward(m, input)
            grad_out = ones_tensor(size(logits)...)
            grad_in = backward(m, grad_out)
            @test grad_in !== nothing
        end
    end

    @testset "MoE" begin
        @testset "router" begin
            router = Router(64, 4, 2)
            input = from_array(reshape(randn(Float32, 1 * 2 * 64), 1, 2, 64))
            weights, indices = forward(router, input)
            @test size(weights) == (2, 2)
            @test length(indices) == 2
            for i in 1:2
                @test length(indices[i]) == 2
            end
            wd = weights.data
            for i in 1:2
                s = wd[i, 1] + wd[i, 2]
                @test abs(s - 1f0) < 0.01f0
            end
        end
        @testset "moe layer shape" begin
            moe = MoELayer(64, 256, 4, 2)
            input = from_array(reshape(randn(Float32, 1 * 2 * 64), 1, 2, 64))
            output = forward(moe, input)
            @test size(output) == size(input)
        end
        @testset "aux loss" begin
            router = Router(64, 4, 2)
            input = from_array(reshape(randn(Float32, 1 * 8 * 64), 1, 8, 64))
            forward(router, input)
            aux = compute_aux_loss(router, 0.01f0)
            @test aux >= 0f0
        end
    end

    @testset "Training" begin
        @testset "cross entropy finite" begin
            logits = from_array(reshape(Float32[1, 2, 3, 2, 1, 0], 1, 2, 3))
            targets = from_array(Float32[2 0])
            targets = reshape_tensor(targets, 1, 2)
            loss = cross_entropy_loss(logits, targets)
            @test isfinite(loss)
            @test loss > 0f0
        end
        @testset "cross entropy grad shape" begin
            logits = from_array(reshape(Float32[1, 2, 3, 2, 1, 0], 1, 2, 3))
            targets = from_array(Float32[2 0])
            targets = reshape_tensor(targets, 1, 2)
            grad = cross_entropy_grad(logits, targets)
            @test size(grad) == size(logits)
            row0 = grad.data[1, 1, :]
            row1 = grad.data[1, 2, :]
            @test abs(sum(row0)) < 1f-4
            @test abs(sum(row1)) < 1f-4
        end
        @testset "train step" begin
            m = tiny_model()
            cfg = default_train_config()
            trainer = Trainer(m, cfg)
            batch, seq_len = 2, 8
            input_data = Float32[Float32(mod(i, 100)) for i in 0:batch*seq_len-1]
            target_data = Float32[Float32(mod(i + 1, 100)) for i in 0:batch*seq_len-1]
            input = from_array(reshape(input_data, batch, seq_len))
            targets = from_array(reshape(target_data, batch, seq_len))
            loss = train_step!(trainer, input, targets)
            @test loss >= 0f0
            @test trainer.step == 1
        end
        @testset "multi step" begin
            m = tiny_model()
            cfg = default_train_config()
            trainer = Trainer(m, cfg)
            batch, seq_len = 1, 4
            input_data = Float32[Float32(mod(i, 100)) for i in 0:batch*seq_len-1]
            target_data = Float32[Float32(mod(i + 1, 100)) for i in 0:batch*seq_len-1]
            input = from_array(reshape(input_data, batch, seq_len))
            targets = from_array(reshape(target_data, batch, seq_len))
            losses = Float32[]
            for _ in 1:5
                push!(losses, train_step!(trainer, input, targets))
            end
            @test trainer.step == 5
            @test all(l -> l >= 0f0, losses)
        end
        @testset "lr schedule" begin
            m = tiny_model()
            cfg = TrainConfig(1f-3, 0.9f0, 0.95f0, 1f-8, 0.1f0, 1f0, 100, 1000, 0.01f0)
            trainer = Trainer(m, cfg)
            trainer.step = 0
            @test get_lr(trainer) == 0f0
            trainer.step = cfg.warmup_steps
            @test abs(get_lr(trainer) - cfg.lr) < 1f-6
            trainer.step = cfg.total_steps * 10
            min_lr = cfg.lr * 0.1f0
            @test get_lr(trainer) >= min_lr - 1f-7
        end
    end

    @testset "Checkpoint" begin
        @testset "storage" begin
            store = CheckpointStorage(true)
            @test store.enabled
            t = ones_tensor(4)
            save!(store, 0, t)
            @test get_checkpoint(store, 0) !== nothing
            @test length(store) == 1
            clear!(store)
            @test length(store) == 0
        end
        @testset "context should_checkpoint" begin
            ctx = CheckpointContext(2)
            @test should_checkpoint(ctx, 0)
            @test !should_checkpoint(ctx, 1)
            @test should_checkpoint(ctx, 2)
            @test !should_checkpoint(ctx, 3)
        end
    end

    @testset "Infrastructure" begin
        @testset "loss scaler dynamic" begin
            scaler = dynamic_loss_scaler()
            @test scaler.current_scale == 65536f0
            @test scale_loss(scaler, 1f0) == 65536f0
            @test unscale_grads(scaler, 65536f0) == 1f0
            @test !should_skip_step(scaler)
        end
        @testset "mixed precision config" begin
            cfg = default_mixed_precision_config()
            @test !cfg.enabled
            @test is_fp32_layer(cfg, "final_norm")
            @test is_fp32_layer(cfg, "lm_head")
            @test !is_fp32_layer(cfg, "attention")
        end
        @testset "master weights" begin
            p1 = ones_tensor(4, 8)
            p2 = ones_tensor(16)
            mw = MasterWeights([p1, p2])
            @test length(mw.weights) == 2
            @test size(mw.weights[1]) == (4, 8)
        end
        @testset "grad clip" begin
            data = fill(100f0, 10)
            t = from_array(reshape(data, 10))
            norm = clip_grad_by_global_norm!(t, 1f0)
            @test norm > 0f0
            sum_sq = sum(x -> x * x, t.data)
            @test sqrt(sum_sq) <= 1f0 + 1f-4
        end
    end
end
