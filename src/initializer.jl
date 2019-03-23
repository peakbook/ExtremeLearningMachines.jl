abstract type Initializers end

struct Uniform <: Initializers
    min::AbstractFloat
    max::AbstractFloat
    ortho::Bool
    function Uniform(;min=-1.0,max=1.0,ortho=false)
        @assert(min<max)
        new(min,max,ortho)
    end
end

function orthonormalize(X::Union{AbstractArray{T},AbstractArray{Complex{T}}}) where {T<:AbstractFloat}
    U,S,V = svd(X)
    return U*transpose(V)
end

function orthonormalize(X::AbstractArray{Quaternion{T}}) where {T<:AbstractFloat}
    Xc = Quaternions.equiv(X)
    U,S,V = svd(Xc)
    return Quaternions.equiv(U*transpose(V))
end

function rand_arr(T::Type{S}, dim::Tuple, init::Uniform) where {S<:AbstractFloat}
    M = rand(T, dim)*T(init.max-init.min) .+ T(init.min)
    return init.ortho ? orthonormalize(M) : M
end

function rand_arr(T::Type{Complex{S}}, dim::Tuple, init::Uniform) where {S<:AbstractFloat}
    M = rand(T, dim)*T(init.max-init.min) .+ T(init.min, init.min)
    return init.ortho ? orthonormalize(M) : M
end

function rand_arr(T::Type{Quaternion{S}}, dim::Tuple, init::Uniform) where {S<:AbstractFloat}
    M = rand(T, dim)*T(init.max-init.min) .+ T(init.min, init.min, init.min, init.min)
    return init.ortho ? orthonormalize(M) : M
end
