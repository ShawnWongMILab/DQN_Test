function dqn = dqnbp(dqn,t,y)
    alpha = 0.01;
    L = numel(dqn.dqnshape);
    current_A = find(y==dqn.Q);
    dqn.delta{L} = [0;0;0;0;0];
    delta_L = (y - t) .* dqn.dotf{L}(dqn.z{L});
    dqn.delta{L}(current_A) = delta_L;
    dqn.J = sum(sum((y - t).^2)) / 2; 
    for l = (L - 1):-1:2
        dqn.delta{l} = (dqn.dotf{l}(dqn.z{l}) .* (dqn.w{l}' * dqn.delta{l+1}));
    end
    
    for l = (L-1):-1:1
        dw = dqn.delta{l+1} * dqn.a{l}';
        dqn.w{l} = dqn.w{l} - alpha * dw;
    end
end