function adj_mat(data)::SparseMatrixCSC
    adjli = data.adjacency_list
    N = length(adjli)
    adjmat = spzeros(N,N)
    for i in 1:N
        for j in adjli[i]
            adjmat[i,j] = 1
        end
    end
    return adjmat
end

function adj_norm(A::SparseMatrixCSC{<:Real, <:Real})
    Ã = A + I
    deg = (sum∘eachcol)(Ã)
    D̃ₛ = spdiagm(deg.^-.5)
    return D̃ₛ*A*D̃ₛ
end

function accuracy(ŷ,y)
    C,N = size(y) 
    ŷ,y = Flux.onecold.((ŷ,y),(1:C,1:C))
    return sum(ŷ .== y)/N
end

function train!(gnn,data;epochs = 500,patience = 100,lr = 5e-3)
    A = adj_mat(data)
    Ã = adj_norm(A)
    X = sparse(data.node_features)
    y = Flux.onehotbatch(data.node_labels,1:data.num_classes)
    
    train_mask = data.train_indices
    test_mask = data.test_indices
    val_mask = data.val_indices
    
    best_test_acc = 0
    best_val_acc = 0
    no_improv = 0
    
    opt = ADAM(lr)
    θ = params(gnn)
    for epoch in 1:epochs
        local loss
        gs = gradient(θ) do
            ŷ = gnn(X,Ã)
            loss = Flux.Losses.crossentropy(ŷ[:,train_mask], y[:,train_mask])
            return loss
        end
        Flux.Optimise.update!(opt, θ, gs)
        
        ŷ = gnn(X,Ã)
        val_acc = accuracy(ŷ[:,val_mask],y[:,val_mask])
        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_test_acc = accuracy(ŷ[:,test_mask],y[:,test_mask])
            no_improv = 0
        else
            no_improv += 1
            no_improv >= patience && break
        end
        if epoch % 10 == 0 
            println("Epoch $epoch | Loss $loss | Val ACC $val_acc | Best Test ACC $best_test_acc")
        end
    end
end