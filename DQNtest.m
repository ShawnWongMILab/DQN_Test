epsilon = 0.3;
gama = 0.98;
max_epoch = 1000;

% MDP

S = eye(18);
A = [1,2,3,4,5];

dqn = dqnsetup();

for epoch = 1:max_epoch
    
    dqn.J
    current_S = S(:,unidrnd(18));
    current_s_Index = find(current_S);
    while current_s_Index ~= 17 && current_s_Index ~= 18
        %fprintf('epoch:%8d\tcurrent_state:%f\n',epoch,current_s_Index)
        dqn = dqnff(dqn,current_S);
        if rand < epsilon
            current_A = A(unidrnd(5));
        else
            [max_Q,current_A]=max(dqn.Q);
        end
        y = dqn.Q(current_A); % output of network
        %y
        [next_S,reward]=executeAction(current_S,current_A);
        next_s_Index = find(next_S);
        if next_s_Index == 17 || next_s_Index == 18
            t = reward;  %target
        else
            dqn = dqnff(dqn,next_S);
            [max_Q,next_A]=max(dqn.Q);
            t = reward + gama * max_Q; %target
        end
        dqn = dqnff(dqn,current_S);
        dqn = dqnbp(dqn,t,y);
       
        current_S = next_S;
        current_s_Index = next_s_Index;
        %fprintf('epoch:%8d\tnext_state:%f\n',epoch,next_s_Index)
    end
end
