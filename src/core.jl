abstract ELMTypes
type Regression <: ELMTypes end
type Classification <: ELMTypes
    nclass::Int
end

type ELM{T<:BaseType}
    W::Matrix{T}
    b::Matrix{T}
    beta::Matrix{T}
    L::Int
    mf::MappingFunctions
    elmtype::ELMTypes

    function ELM(L::Int;
                 mf::MappingFunctions=Sigmoid(),
                 elmtype::ELMTypes=Regression())
        net = new()
        net.L = L
        net.mf = mf
        net.elmtype = elmtype
        return net
    end
end

function train!{T<:BaseType, S<:Union{BaseType,Int}}(net::ELM{T}, X::Matrix{T}, Y::Matrix{S};
                             tm::TrainingMethods=Batch(),
                             init::Initializers=Uniform())
    assert(size(Y,2) == size(X,2))

    net.W = rand_arr(T, (net.L, size(X,1)), init)
    net.b = rand_arr(T, (net.L, 1), init)

    X, Y = preprocess(X, Y, net.elmtype)
    net.beta = train_core(net.W, net.b, X, Y, net.mf, tm)

    return net
end

function predict{T<:BaseType}(net::ELM{T}, X::Matrix{T})
    H = mapping(net.W, net.b, X, net.mf)
    Y = net.beta*H
    return postprocess(Y, net.elmtype)
end

function preprocess{T<:BaseType}(X::Matrix{T}, Y::Matrix{T}, elmtype::Regression)
    return X, Y
end

function postprocess{T<:BaseType}(Y::Matrix{T}, elmtype::Regression)
    return Y
end

function preprocess{T<:BaseType, S<:Int}(X::Matrix{T}, Y::Matrix{S}, elmtype::Classification)
    assert(maximum(Y) < elmtype.nclass, "Class labels must be in {0,1,...,nclass-1}.")
    if elmtype.nclass == 2
        Yc = zeros(T, 1, size(Y,2))
        for i=1:size(Y,2)
            Yc[1,i] = Y[1,i] > zero(S) ? one(T) : -one(T)
        end
    else
        Yc = zeros(T, elmtype.nclass, size(Y,2))
        for i=1:size(Y,2)
            Yc[Y[1,i]+one(S),i] = one(T)
        end
    end
    return X, Yc
end

function postprocess{T<:BaseType}(Yc::Matrix{T}, elmtype::Classification)
    Y = zeros(Int, 1, size(Yc,2))
    if elmtype.nclass == 2
        for i=1:size(Yc,2)
            Y[1,i] = real(Yc[i]) > zero(Int) ? one(Int) : zero(Int)
        end
    else
        for i=1:size(Yc,2)
            Y[1,i] = indmax(real(Yc[:,i])) - one(Int)
        end
    end
    return Y
end

