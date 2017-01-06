abstract MappingFunctions

type Sigmoid   <: MappingFunctions end
type HardLimit <: MappingFunctions end
type ReLU      <: MappingFunctions end
type Cosine    <: MappingFunctions end
type Qubit     <: MappingFunctions end
type HyperbolicTangent <: MappingFunctions end


function mapping{T<:BaseType}(W::Matrix{T}, b::Matrix{T}, X::Matrix{T}, mf::MappingFunctions)
    return map(x->mapfunc(x, mf), W*X .+ b)
end

function mapping{T<:BaseType}(W::Vector{T}, b::T, X::Vector{T}, mf::MappingFunctions)
    return map(x->mapfunc(x, mf), dot(conj(W),X) + b)
end


function mapfunc{T<:AbstractFloat}(x::T, mf::Sigmoid)
    return one(T)/(one(T)+exp(-x))
end

function mapfunc{T<:AbstractFloat}(x::T, mf::HardLimit)
    return x >= zero(T) ? one(T) : zero(T)
end

function mapfunc{T<:AbstractFloat}(x::T, mf::ReLU)
    return x >= zero(T) ? x : zero(T)
end

function mapfunc{T<:AbstractFloat}(x::T, mf::Cosine)
    return cos(x)
end

function mapfunc{T<:AbstractFloat}(x::T, mf::Qubit)
    return sin(0.5*pi*x)^2
end

function mapfunc{T<:AbstractFloat}(x::T, mf::HyperbolicTangent)
    v = exp(-x)
    return (one(T)-v)/(one(T)+v)
end

# split-type functions
for mf in (:Sigmoid, :HardLimit, :ReLU, :Cosine, :Qubit, :HyperbolicTangent)
    @eval begin
        type $(Symbol(mf,:Split)) <: MappingFunctions end
        export $(Symbol(mf,:Split)) 

        function mapfunc{T<:AbstractFloat}(x::T, mf::$(Symbol(mf,:Split)))
            return mapfunc(x, $(mf)())
        end

        function mapfunc{T<:AbstractFloat}(x::Complex{T}, mf::$(Symbol(mf,:Split)))
            return complex(mapfunc(real(x), $(mf)()),
                           mapfunc(imag(x), $(mf)()))
        end

        function mapfunc{T<:AbstractFloat}(x::Quaternion{T},  mf::$(Symbol(mf,:Split)))
            return quaternion(mapfunc(real(x),  $(mf)()),
                              mapfunc(imagi(x), $(mf)()),
                              mapfunc(imagj(x), $(mf)()),
                              mapfunc(imagk(x), $(mf)()))
        end
    end
end

