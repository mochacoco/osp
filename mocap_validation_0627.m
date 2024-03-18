clear all;
close all;
clc;

% Generate Dataset

load('mocap_X_3.mat');
load('mocap_Y_3.mat');
teach_include = 1:13414;
% 
% X = load('DFM_X.mat').X;
% Y = load('DFM_Y.mat').Y;
% teach_include = 1:22680;

X = X(teach_include,:);
Y = Y(teach_include,:);
X = (X-mean(X))./std(X);
Y = (Y-mean(Y))./std(Y);


designate = 0;
interest =[1,2,3,4,5,6];
gamma = 0.1; %5
m = 36;
s_vector = 1:m;
Ns = 200;%500;
Niter = 200;
%m0 = m;
m0 = 5;
tau = ones(m,1)./m.*m0;

tau_log = zeros(Niter,m);
T = zeros(Ns,1);
Nx = 128;
lamb_iter = zeros(Niter,1);
logprob_iter = zeros(Niter,1);
loadsigma = 2;
sigX = std(X);
sigma_n = 0.1;
sigma_g = 1;
betaA = ones(m,1);
betaB = ones(m,1);
betasample = zeros(Ns, m);
drchnumber = ones(1,m);
gama = m;
gamb = 1;

for iter = 1:1:Niter
    %m0 = m0_target;
    tau_prev = tau;
    z = zeros(Ns,m);
    T = zeros(Ns,1);
    z_update = zeros(m,1);
    loggrad = zeros(m+2,1);
    m0_samp = zeros(Ns,1);
    for i=1:1:Ns
        betasample(i,:) = betarnd(betaA, betaB)';
%         single_samp = min(max(round(gamrnd(gama, 1./gamb)),1),m);
%         m0_samp(i,1) = single_samp;
        %m0 = single_samp;
        for j=1:1:m0
            [~, argmaxval] = max(betasample(i,:));
            z(i,argmaxval) = 1;
            betasample(i,argmaxval) = 0;
        end
        

        samp = randi([1,length(X)],1,Nx);
        X_sample = X(samp,:)*diag(z(i,:)); % diag(z(i,:)) = (20,20) , X(samp,:) = (200,20)
        Y_sample = Y(samp,interest);
        logprob = 0;
        
        [K_sample, ~] = kernel(X_sample,sigX,sigma_n,sigma_g,1);
        L = chol(K_sample);
        Lalpha = L\(L'\Y_sample);
        for kk = 1:1:length(interest)
            logprob = logprob - 0.5*Y_sample(:,kk)'*Lalpha(:,kk) - sum(log(diag(L))); %- Nx/2*log(2*pi);
            %logprob = logprob - norm(Y_sample(:,kk) - K_sample*Lalpha(:,kk)).^2;
        end
        
        %logprob = -norm(Y_sample - K_sample*Lalpha).^2;

        T(i,1) = logprob;
        
    end

%     for jj = 1:1:Ns
%         drchnumber(1,m0_samp(jj,1)) = drchnumber(1,m0_samp(jj,1)) + exp(T(jj,1)).*double(T(jj,1) - mean(T) >= 0);
%     end
%     drchnumber = drchnumber * 0.99;
%     scalefactor = (1.00)^(iter);
%     gamb = gamb.*0.95 + 1.*scalefactor;
    [~, maxvalT] = max(T);
    [~, minvalT] = min(T);
%     gama = gama.*0.95 + m0_samp(maxvalT,1).*scalefactor;
%     lamb_iter(iter,1) = gama./gamb;

%     
     logprob_iter(iter,1) = mean(T);
%     betaA = betaA.*0.999 + z(maxvalT,:)';
%     betaB = betaB.*0.999 + z(minvalT,:)';


    T = T - mean(T);
    %betaA = betaA.*0.95 + z' * (1.*double(T >= 0));
    %betaB = betaB.*0.95 + z' * (1.*double(T <= 0));
if norm((double(T >= std(T)))) ~= 0
    betaA = betaA + z' * (T.*double(T >= std(T)))./norm((T.*double(T >= std(T)))).*0.35;
end
if norm((double(T <= std(T)))) ~= 0
    betaB = betaB + z' * (-T.*double(T <= -std(T)))./norm(-T.*double(T <= -std(T))).*0.35;
end

    betaA = betaA * 0.99;
    betaB = betaB * 0.99;
    

    tau = betaA./(betaA + betaB);
    tau_log(iter,:) = betaA./(betaA + betaB);

    

%     samp = [1:round(length(X)/Nx):length(X)];
%     X_sample = X(samp,:); % diag(z(i,:)) = (20,20) , X(samp,:) = (200,20)
%     Y_sample = Y(samp,interest);
%     loggrad = zeros(m,1);
%     [K_sample, K_grad] = kernel(X_sample,sigX,sigma_n,sigma_g,0);
%     L = chol(K_sample);
%     Lalpha = L\(L'\Y_sample);
%     for k = 1:1:size(Y_sample,2)
%         aa = (Lalpha(:,k)*Lalpha(:,k)')- (1/size(Y_sample,2)).*L\(L'\ones(length(samp),1));
%         for kk=1:1:m
%             loggrad(kk,1) = loggrad(kk,1) + 0.5*trace(aa*K_grad(:,:,kk));
%         end
%     end

    %sigX = sigX + (0.001).*loggrad(1:m,1)';

    subplot(5,4,[1:2,5:6,9:10]);
    xlabel('Iteration #');
    ylabel('Tau');
    hold on;
    axis([0 inf 0 1]);
    for k=1:1:m
        plot(iter,tau(k,1),'.','MarkerEdgeColor',[k/m,0,0]);
        plot(1:iter,tau_log(1:iter,k),'Color',[k/m,0,0]);
        %mytext(k) = text(iter,tau(k,1),'\leftarrow'+string(k));
    end
    if length(tau(tau==1)) ~= length(tau_prev(tau_prev==1))
        continue;
%         if length(tau(tau==1)) == m0_target
%             text(iter,1,'\leftarrow'+string(s_vector(tau==1)),'color','r');
%         else
%             text(iter,1,'\leftarrow'+string(s_vector(tau==1)),'color','b');
%         end
    else
        text(iter,tau_log(iter,mod(iter,m)+1),'\leftarrow'+string(mod(iter,m)+1),'color','k')
    end

    hold off;
    drawnow;
    
    
    subplot(5,4,[3:4,7:8,11:12]);
    
    
    for k=1:1:m
        if k == 2
            hold on;
        end
        x=0:0.01:1;
        y = betapdf(x, betaA(k,1),betaB(k,1));
        plot(y,x,'Color',[k/m,0,0]);
        mean1 = betaA(k,1)./(betaA(k,1)+betaB(k,1));
        text(betapdf(mean1, betaA(k,1),betaB(k,1)),mean1,'\leftarrow'+string(k),'color','k')
    end
    xlabel('Prob.');
    ylabel('Tau');
    hold off;
    drawnow;

    subplot(5,4,[13:16]);
    if iter > 2
        plot(2:iter,logprob_iter(2:iter,1),'r*');
        hold on;
        xlabel('Iteration #');
        ylabel('Log Likelihood');
        hold off;
        drawnow;
    end
    subplot(5,4,[17:20]);
    bar(tau(:,1));
%     subplot(5,4,[17:20]);
% %     bar(drchnumber);
%     plot(1:iter,lamb_iter(1:iter,1),'r*');
%     hold on;
%     xlabel('m0');
%     ylabel('Likelihood');
%     hold off;
%     drawnow;
end



function [K,Kp] = kernel(X,sig,sig_n,sig_g,digit)
    K = zeros(length(X),length(X));
    Kp = zeros(length(X),length(X),length(sig)+2);
    eps = (rand*1e-5) + 1e-5;
    if digit == 0
        for i=1:1:length(X)
            for j=1:1:length(X)
                K(i,j) = sig_g^2*exp(-norm(diag(1./sig)*(X(i,:)-X(j,:))')^2) + sig_n*max(1-abs(i-j),0);%norm(X(i,:)-X(j,:))^2)/var);
                for k=1:1:length(sig)
                    sigplus = sig;
                    sigplus(k) = sigplus(k) + eps;
                    Kp(i,j,k) = (sig_g^2*exp(-norm(diag(1./sigplus)*(X(i,:)-X(j,:))')^2) + sig_n*max(1-abs(i-j),0) - K(i,j))./eps;
                end
                Kp(i,j,length(sig)+1) = (sig_g^2*exp(-norm(diag(1./sigplus)*(X(i,:)-X(j,:))')^2) + (sig_n+eps)*max(1-abs(i-j),0)- K(i,j))./eps;
                Kp(i,j,length(sig)+2) = ((sig_g+eps)^2*exp(-norm(diag(1./sig)*(X(i,:)-X(j,:))')^2) + sig_n*max(1-abs(i-j),0) - K(i,j))./eps;
            end
        end
    else
        for i=1:1:length(X)
            for j=1:1:length(X)
                K(i,j) = sig_g^2*exp(-norm(diag(1./sig)*(X(i,:)-X(j,:))')^2) + sig_n*max(1-abs(i-j),0);%norm(X(i,:)-X(j,:))^2)/var);
            end
        end
    end
end


function r = drchrnd(a, n)
    p = length(a);
    r = gamrnd(repmat(a,n,1),1,n,p);
    r = r./repmat(sum(r,2),1,p);
end