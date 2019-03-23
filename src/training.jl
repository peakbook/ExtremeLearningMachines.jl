abstract type TrainingMethods end

struct Batch <: TrainingMethods end
struct BatchL2 <: TrainingMethods
    lambda::AbstractFloat
end
struct BatchL1 <: TrainingMethods
    lambda::AbstractFloat
end

struct BatchCM <: TrainingMethods end
struct BatchCML2 <: TrainingMethods
    lambda::AbstractFloat
end

function train_core(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                    mf::MappingFunctions, tm::Batch) where {T<:BaseType}
    H = mapping(W, b, X, mf)
    return Y*pinv(H);
end

function train_core(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                    mf::MappingFunctions, tm::BatchL2) where {T<:BaseType}
    H = mapping(W, b, X, mf)
    L = size(H,1)
    N = size(H,2)
    tH = ctranspose(H)
    if N > L
        return Y*tH*pinv(H*tH + tm.lambda*eye(T,L))
    else
        return Y*(pinv(tH*H + tm.lambda*eye(T,N))*tH)
    end
end

function train_core(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                    mf::MappingFunctions, tm::BatchL1) where {T<:AbstractFloat}
    H = mapping(W, b, X, mf)
    beta = Variable(size(Y,1), size(W,1))
    problem = minimize(0.5*norm(beta*H-Y,2)^2 + tm.lambda * norm(beta,1))
    solve!(problem, ECOSSolver(verbose=false))
    return beta.value
end

function train_core(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                    mf::MappingFunctions, tm::BatchCM) where {T<:BaseType}
    A, B = cormat(W, b, X, Y, mf)
    return B*pinv(A)'
end

function train_core(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                    mf::MappingFunctions, tm::BatchCML2) where {T<:BaseType}
    A, B = cormat(W, b, X, Y, mf)
    return B*pinv(A+tm.lambda*eye(T,size(A,1)))'
end

function cormat(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, Y::Matrix{T},
                mf::MappingFunctions) where {T<:BaseType}
    P = size(X,2)
    Nh = size(W, 1)
    No = size(Y, 1)
    Hj = zeros(T, Nh, 1)
    A = zeros(T, Nh, Nh)
    B = zeros(T, No, Nh)
    for i=1:P
        for j=1:Nh
            Hj[j] = mapping(W[j,:], b[j], X[:,i], mf);
        end
        tHj = ctranspose(Hj)
        A += Hj * tHj
        B += Y[:,i] * tHj
    end

    return A, B
end
