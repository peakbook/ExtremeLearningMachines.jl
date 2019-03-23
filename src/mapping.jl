abstract type MappingFunctions end

struct Sigmoid   <: MappingFunctions end
struct HardLimit <: MappingFunctions end
struct ReLU      <: MappingFunctions end
struct Cosine    <: MappingFunctions end
struct Qubit     <: MappingFunctions end
struct HyperbolicTangent <: MappingFunctions end


function mapping(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, mf::MappingFunctions) where {T<:BaseType}
    return mf.(W*X .+ b)
end

function mapping(W::Vector{T}, b::T, X::Vector{T}, mf::MappingFunctions) where {T<:BaseType}
    return mf.(dot(conj(W),X) + b)
end


function mapfunc(x::T, mf::Sigmoid) where {T<:AbstractFloat}
    return one(T)/(one(T)+exp(-x))
end

function mapfunc(x::T, mf::HardLimit) where {T<:AbstractFloat}
    return x >= zero(T) ? one(T) : zero(T)
end

function mapfunc(x::T, mf::ReLU) where {T<:AbstractFloat}
    return x >= zero(T) ? x : zero(T)
end

function mapfunc(x::T, mf::Cosine) where {T<:AbstractFloat}
    return cos(x)
end

function mapfunc(x::T, mf::Qubit) where {T<:AbstractFloat}
    return sin(0.5*pi*x)^2
end

function mapfunc(x::T, mf::HyperbolicTangent) where {T<:AbstractFloat}
    v = exp(-x)
    return (one(T)-v)/(one(T)+v)
end

# split-type functions
for mf in (:Sigmoid, :HardLimit, :ReLU, :Cosine, :Qubit, :HyperbolicTangent)
    @eval begin
        struct $(Symbol(mf,:Split)) <: MappingFunctions end
        export $(Symbol(mf,:Split))

        function mapfunc(x::T, mf::$(Symbol(mf,:Split))) where {T<:AbstractFloat}
            return mapfunc(x, $(mf)())
        end

        function mapfunc(x::Complex{T}, mf::$(Symbol(mf,:Split))) where {T<:AbstractFloat}
            return complex(mapfunc(real(x), $(mf)()),
                           mapfunc(imag(x), $(mf)()))
        end

        function mapfunc(x::Quaternion{T},  mf::$(Symbol(mf,:Split))) where {T<:AbstractFloat}
            return quaternion(mapfunc(real(x),  $(mf)()),
                              mapfunc(imagi(x), $(mf)()),
                              mapfunc(imagj(x), $(mf)()),
                              mapfunc(imagk(x), $(mf)()))
        end
    end
end
