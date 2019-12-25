module TriMine
    using SpecialFunctions
    using Distributions
    using DelimitedFiles
    export infer

    struct Params
        X::Array{Int64, 3}  # tensor
        k::Int64  # of topics
        u::Int64  # of objects
        v::Int64  # of actors
        n::Int64  # duration
        amax::Float64
        bmax::Float64
        gmax::Float64
    end

    mutable struct Vars
        a::Float64  # alpha
        b::Float64  # beta
        g::Float64  # gamma
        Z::Array{Int64, 3}
        O::Array{Float64, 2}  # Object matrix
        A::Array{Float64, 2}  # Actor  matrix
        C::Array{Float64, 2}  # Time   matrix
    end

    mutable struct Counter
        Nk::Array{Int64, 1}
        Nu::Array{Int64, 1}
        Nku::Array{Int64, 2}
        Nkv::Array{Int64, 2}
        Nkn::Array{Int64, 2}
    end

    function infer(prm::Params, var::Vars, cnt::Counter)
        # cnt.Nu = sum(reshape(prm.X, prm.u, prm.v * prm.n), dims=2)[:, 1]
        for i = 1:prm.u
            cnt.Nu[i] = sum(prm.X[i, :, :])
        end
        gibbs_sampling(prm, var, cnt) 
        compute_factors(prm, var, cnt)
        update_alpha(prm, var, cnt)
        update_beta(prm, var, cnt)
        update_gamma(prm, var, cnt)
        # println(var.a)
        # println(var.b)
        # println(var.g)
    end

    function gibbs_sampling(prm::Params, var::Vars, cnt::Counter)
        #= Equation (1) =#

        for t = 1:prm.n
            for i = 1:prm.u
                for j = 1:prm.v
                    x = prm.X[i, j, t]
                    if x > 0
                        z = var.Z[i, j, t]

                        if z > 0
                            cnt.Nk[z] -= x
                            cnt.Nku[z, i] -= x
                            cnt.Nkv[z, j] -= x
                            cnt.Nkn[z, t] -= x
                        end

                        p = zeros(prm.k)

                        for r = 1:prm.k
                            O = (cnt.Nku[r, i] + var.a) / (cnt.Nu[i] + var.a * prm.k)
                            A = (cnt.Nkv[r, j] + var.b) / (cnt.Nk[r] + var.b * prm.v)
                            C = (cnt.Nkn[r, t] + var.g) / (cnt.Nk[r] + var.g * prm.n)
                            p[r] = O * A * C
                        end

                        p /= sum(p)
                        z = argmax(rand(Multinomial(1, p), 1)[:, 1])
                        var.Z[i, j, t] = z
                        cnt.Nk[z] += x
                        cnt.Nku[z, i] += x
                        cnt.Nkv[z, j] += x
                        cnt.Nkn[z, t] += x
                    end
                end
            end
        end
    end

    function compute_loglikelihood(prm::Params, var::Vars, cnt::Counter)
        llh = loggamma(var.a * prm.k) - prm.k * loggamma(var.a)
        llh += loggamma(var.b * prm.k) - prm.k * loggamma(var.b)
        llh += loggamma(var.g * prm.k) - prm.k * loggamma(var.g)
        for i = 1:prm.k
            val = 0
            for j = 1:prm.u
                val += log(var.O[j, i])
            end
            llh += (var.a - 1) * val / prm.u
            val = 0
            for j = 1:prm.v
                val += log(var.A[j, i])
            end
            llh += (var.b - 1) * val / prm.v
            val = 0
            for j = 1:prm.n
                val += log(var.C[j, i])
            end
            llh += (var.g - 1) * val / prm.n
        end
        return llh
    end

    function compute_factors(prm::Params, var::Vars, cnt::Counter)
        for i = 1:prm.k
            for j = 1:prm.u
                var.O[j, i] = cnt.Nku[i, j] + var.a
                var.O[j, i] /= cnt.Nu[j] + var.a * prm.k
            end
            for j = 1:prm.v
                var.A[j, i] = cnt.Nkv[i, j] + var.b
                var.A[j, i] /= cnt.Nk[i] + prm.v * var.b
            end
            for j = 1:prm.n
                var.C[j, i] = cnt.Nkn[i, j] + var.g
                var.C[j, i] /= cnt.Nk[i] + prm.n * var.g
            end
        end
    end

    function update_alpha(prm::Params, var::Vars, cnt::Counter)
        n = -1 * prm.u * prm.k * digamma(var.a)
        d = -1 * prm.u * prm.k * digamma(var.a * prm.u)
        for i = 1:prm.k
            d += prm.u * digamma(cnt.Nk[i] + var.a * prm.u)
            for j = 1:prm.u
                n += digamma(cnt.Nku[i, j] + var.a)
            end
        end
        var.a *= n / d
        if var.a > prm.amax
            var.a = prm.amax
        end
    end

    function update_beta(prm::Params, var::Vars, cnt::Counter)
        n = -1 * prm.k * prm.v * digamma(var.b)
        d = -1 * prm.k * prm.v * digamma(var.b * prm.v)
        for i = 1:prm.k
            d += prm.v * digamma(cnt.Nk[i] + var.b * prm.v)
            for j = 1:prm.v
                n += digamma(cnt.Nkv[i, j] + var.b)
            end
        end
        var.b *= n / d
        if var.b > prm.bmax
            var.b = prm.bmax
        end        
    end

    function update_gamma(prm::Params, var::Vars, cnt::Counter)
        n = -1 * prm.k * prm.n * digamma(var.g)
        d = -1 * prm.k * prm.n * digamma(var.g * prm.n)
        for i = 1:prm.k
            d += prm.n * digamma(cnt.Nk[i] + var.g * prm.n)
            for j = 1:prm.n
                n += digamma(cnt.Nkn[i, j] + var.g)
            end
        end
        var.g *= n / d
        if var.g > prm.gmax
            var.g = prm.gmax
        end
    end

    function online_learning(prm::Params, var::Vars, cnt::Counter, niter::Int64)
        for i = 1:niter
            infer(prm, var, cnt)
            llh = compute_loglikelihood(prm, var, cnt)
            println(llh)
        end
    end

    function save_model(prm::Params, var::Vars, cnt::Counter)
        writedlm("out/O.txt", var.O, ",")
        writedlm("out/A.txt", var.A, ",")
        writedlm("out/C.txt", var.C, ",")
    end
end
