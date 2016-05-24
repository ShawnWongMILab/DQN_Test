function dqn = dqnff(dqn,x)
    L = numel(dqn.dqnshape);
    dqn.a{1} = x;
    for l = 1:(L-1)
        dqn.z{l+1} = dqn.w{l} * dqn.a{l};
        dqn.a{l+1} = dqn.f{l+1}(dqn.z{l+1});
    end 
    dqn.Q = dqn.a{L};
end

