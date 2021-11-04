struct GCN²
    conv1::GraphConv
    conv2::GraphConv
    dropout::Float64
    ρ
end

function GCN²(in_feats::Integer, h_feats::Integer, out_feats::Integer;σ=identity,ρ=relu,dropout=0,bias=true)
    conv1 = GraphConv(in_feats,h_feats;σ=σ,bias=bias)
    conv2 = GraphConv(h_feats,out_feats;σ=σ,bias=bias)
    return GCN²(conv1,conv2,dropout,ρ)
end

function (m::GCN²)(X,A)
    X = Dropout(m.dropout)(X)
    X = m.conv1(X,A)
    X = m.ρ.(X)
    X = Dropout(m.dropout)(X)
    X = m.conv2(X,A)
    return softmax(X)
end

Flux.@functor GCN²
Flux.params(m::GCN²) = params(m.conv1,m.conv2)