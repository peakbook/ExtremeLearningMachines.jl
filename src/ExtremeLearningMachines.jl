module ExtremeLearningMachines

using JLD
using Quaternions
using Convex
using ECOS

export ELM
export Regression, Classification
export train!, predict
export Sigmoid, HardLimit, ReLU, Cosine, Qubit, HyperbolicTangent
export Batch, BatchCM, BatchL2, BatchCML2, BatchL1
export Uniform
export save, load

include("common.jl")
include("mapping.jl")
include("initializer.jl")
include("training.jl")
include("core.jl")
include("io.jl")

end # module
