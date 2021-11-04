struct GraphConv
    linear
end

GraphConv(in_feats::Integer, out_feats::Integer;σ=identity,bias=true) = 
    GraphConv(Dense(in_feats,out_feats,σ;bias=bias))

(m::GraphConv)(X,A) = m.linear(X)*A

Flux.@functor GraphConv
Flux.params(m::GraphConv) = params(m.linear)