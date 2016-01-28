%% Initialize Parameters

clc
clear

alpha = 0;    %coefficient for block fading model
beta = 0.8^2;  %attenuation loss from non-direct antennas
n0 = 10^(-2);    %noise variance

iternums = 1:50; % number of iterations
N_Realizations = 1000;

C1 = zeros(N_Realizations, length(iternums));
C2 = zeros(N_Realizations, length(iternums));
C3 = zeros(N_Realizations, length(iternums));

%% Start Loop
for Realization = 1 : N_Realizations
    Realization
        
    %% Random Channels
    H11 = (randn(2,2)+1i*randn(2,2))/sqrt(2);
    H22 = (randn(2,2)+1i*randn(2,2))/sqrt(2);
    H33 = (randn(2,2)+1i*randn(2,2))/sqrt(2);
    
    H12 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta); 
    H13 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta); 
    H21 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta);
    H23 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta); 
    H31 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta); 
    H32 = (randn(2,2)+1i*randn(2,2))/sqrt(2/beta);
    
    %Backward Channel
    Z11 = H11';
    Z22 = H22';
    Z33 = H33';
    
    Z12 = H21';
    Z13 = H31';
    Z21 = H12';
    Z23 = H32';
    Z31 = H13';
    Z32 = H23';   
    
    %% one iteration per block
    g11 = rand(2, 1) + 1i*rand(2, 1); 
    g12 = rand(2, 1) + 1i*rand(2, 1); 
    g13 = rand(2, 1) + 1i*rand(2, 1); 
    g21 = rand(2, 1) + 1i*rand(2, 1);
    g22 = rand(2, 1) + 1i*rand(2, 1);
    g23 = rand(2, 1) + 1i*rand(2, 1);
    g31 = rand(2, 1) + 1i*rand(2, 1);
    g32 = rand(2, 1) + 1i*rand(2, 1);
    g33 = rand(2, 1) + 1i*rand(2, 1);
    M1 = sqrt(norm(g11)^2+norm(g12)^2+norm(g13)^2);
    g11 = g11/M1;
    g12 = g12/M1;
    g13 = g13/M1;

    M2 = sqrt(norm(g21)^2+norm(g22)^2+norm(g23)^2);
    g21 = g21/M2;
    g22 = g22/M2;
    g23 = g23/M2;

    M3 = sqrt(norm(g31)^2+norm(g32)^2+norm(g33)^2);
    g31 = g31/M3;
    g32 = g32/M3;
    g33 = g33/M3;
 
    v11 = zeros(2, 1); 
    v12 = zeros(2, 1);
    v13 = zeros(2, 1); 
    v21 = zeros(2, 1); 
    v22 = zeros(2, 1);
    v23 = zeros(2, 1); 
    v31 = zeros(2, 1); 
    v32 = zeros(2, 1); 
    v33 = zeros(2, 1); 
    
    for numiters = 1:length(iternums)

        %% bi-directional training
            
            %%Backward Training: sudo-LS Algorithm
            %abs(error)<2*10^(-4)
            [v11, v12, v13, v21, v22, v23, v31, v32, v33] = MSE(Z11, Z12, Z13, Z21, Z22, Z23, Z31, Z32, Z33, g11, g12, g13, g21, g22, g23, g31, g32, g33, n0);
            %[v11, v12, v13]
            %[v21, v22, v23]
            %[v31, v32, v33]
            %Power = [norm(v11)^2+norm(v12)^2+norm(v13)^2 norm(v21)^2+norm(v22)^2+norm(v23)^2 norm(v31)^2+norm(v32)^2+norm(v33)^2]
            
            %%Forward Training: LS Algorithm
            [g11, g12, g13, g21, g22, g23, g31, g32, g33] = MSE(H11, H12, H13, H21, H22, H23, H31, H32, H33, v11, v12, v13, v21, v22, v23, v31, v32, v33, n0);
            %norm(g1)^2
            %norm(g2)^2
            %Power = [norm(g11)^2+norm(g12)^2+norm(g13)^2 norm(g21)^2+norm(g22)^2+norm(g23)^2 norm(g31)^2+norm(g32)^2+norm(g33)^2]

            %{
            g12 = zeros(2, 1);
            g13 = zeros(2, 1); 
            g21 = zeros(2, 1); 
            g23 = zeros(2, 1); 
            g31 = zeros(2, 1); 
            g32 = zeros(2, 1);
            %}
          


            SINR1 = norm(g11'*(H11*v11+H12*v21+H13*v31))^2/(norm(g11'*(H11*v12+H12*v22+H13*v32))^2+norm(g11'*(H11*v13+H12*v23+H13*v33))^2+n0*g11'*g11);
            SINR2 = norm(g22'*(H21*v12+H22*v22+H23*v32))^2/(norm(g22'*(H21*v11+H22*v21+H23*v31))^2+norm(g22'*(H21*v13+H22*v23+H23*v33))^2+n0*g22'*g22);
            SINR3 = norm(g33'*(H31*v13+H32*v23+H33*v33))^2/(norm(g33'*(H31*v12+H32*v22+H33*v32))^2+norm(g33'*(H31*v12+H32*v22+H33*v32))^2+n0*g33'*g33);
            C1(Realization, numiters) = abs(log2(1+SINR1));
            C2(Realization, numiters) = abs(log2(1+SINR2));
            C3(Realization, numiters) = abs(log2(1+SINR3));
            end           
    
end




%% Plot C(bits/channel)
%figure
%hold on

p1=plot(iternums, mean(C1)+mean(C2)+mean(C3),'o');

axis([1 numiters 0 20])

xlabel('Number of iterations')
ylabel('C(bits/channel)')
title('Complex Receivers')
%}
