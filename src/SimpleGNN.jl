module SimpleGNN

using Flux, SparseArrays
using LinearAlgebra:I

export adj_mat,adj_norm,accuracy,train!

include("utils.jl")

export GraphConv

include("layers/conv.jl")

include("models/Models.jl")

end
