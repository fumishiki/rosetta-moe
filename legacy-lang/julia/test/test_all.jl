using Test, Lux, Random, MLCore

@testset "MLCore" begin
    rng = Random.default_rng()
    c = Config(256, 128, 512, 4, 1, 32, 4, 2, 2, 64, 1e4, 16) # Tiny config

    @testset "Prims" begin
        x = randn(Float32, 10, 2)
        @test size(first(Derf()(x, nothing, nothing))) == size(x)
    end

    @testset "Layers" begin
        for (L, sz_in) in [(BLTInput, (10, 2)), (MQAAttention, (c.hidden, 10, 2))]
            l = L(c); ps, st = Lux.setup(rng, l)
            y, _ = l(sz_in == (10, 2) ? rand(0:255, 10, 2) : randn(Float32, sz_in...), ps, st)
            @test size(y, 1) == c.hidden
        end
    end

    @testset "Model" begin
        m = BLTRecurrentMoE(c)
        ps, st = Lux.setup(rng, m)
        y, _ = m(rand(0:255, 10, 2), ps, st)
        @test size(y) == (c.vocab, 10, 2)
    end
end
