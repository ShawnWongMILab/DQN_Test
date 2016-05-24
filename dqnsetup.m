function dqn = dqnsetup()
    % definiation
    dqn.sigmoid = @(z) 1 ./ (1 + exp(-z));
    dqn.dotsigmoid = @(z) sigmoid(z) .* (1 - sigmoid(z));
    dqn.relu = @(z) max(0, z);
    dqn.dotrelu = @(z) double(z > 0); % double will convert false to 0 and true to 1
    dqn.linear = @(z) z;
    dqn.dotlinear = @(z) 1;
    
    dqn.dqnshape = [18 32 5];
    dqn.f = {[], dqn.relu, dqn.linear};
    dqn.dotf = {[],dqn.dotrelu,dqn.dotlinear};
    dqn.batch_size = 100;
    dqn.alpha = 0.8;
    dqn.max_epoch = 100; 
    L = numel(dqn.dqnshape);
    for l = 1:(L - 1)
        dqn.w{l} = rand(dqn.dqnshape(l + 1), dqn.dqnshape(l)) * 0.2 - 0.1;
    end
    dqn.J = 0
end