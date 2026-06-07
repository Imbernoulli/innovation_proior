% Code to reproduce Figure 8: Cylinder flow example
clear; close all; clc

datpath = '../DATA/';
figpath = '../figures/';

load([datpath,'ALL.mat']);


X = VORTALL;
meanvort = mean(X,2);
X = X-repmat(meanvort, 1,size(X,2)); % mean centered data

% 64 images of each person
% seed random number generator for predictable sequence
rng(729); 

trainIdx = 1:100;

Iord = 1:size(X,2);
testIdx = Iord(~ismember(Iord,trainIdx));
XTrain = X(:,trainIdx);
[Psi,S,V] = svd(XTrain,'econ');

[m,n] = size(XTrain);


semilogy(diag(S)/sum(diag(S)),'-');
set(gca,'XTickLabelRotation',45,'YTick',logspace(-8,0,9));
grid on

Y = X(:,testIdx);

% target ranks
var = linspace(0,0.5,20);

%Eopt = zeros(100,length(R));
Eqr = zeros(50,length(var));
Eproj = zeros(50,length(var));
Edeim = zeros(50,length(var));
Edat = zeros(50,length(var));
Esspor = zeros(50,length(var));

rng(729);


%r = 27;
r=40;
% QR sensor selection
[~,~,pivot] = qr(Psi(:,1:r)','vector');
pivot = pivot(1:r);

% DEIM sensor selection
gamma = deim(Psi(:,1:r));
    
% oversampled QR
[xx,yy] = meshgrid(1:5:199,1:5:449);

dgrid = sub2ind([199 449],xx,yy);
%[dgrid,IG,IM] = intersect(dgrid, find(mask==1)); 

%display_sensors(mask,IM,'r');

[~,~,piv] = qr(Psi(dgrid,1:r)*Psi(dgrid,1:r)','vector');
opiv = piv(1:2*r); 
opiv = dgrid(opiv);

% Data matrix QR
[~,~,dpiv] = qr(XTrain(:,1:2*r)','vector');
dpiv = dpiv(1:2*r);

%% Noise comparison: POD vs. DEIM vs. QDEIM

for k = 1:length(var)
    %noise = v*randn(m,n);
                        
    ct = 1;
    for ii =1:50
        
        x = Y(:,ii);
        xnoise = x+var(k)*randn(size(x));
        
        % Projection onto r eigenmodes
        xproj = Psi(:,1:r)*(Psi(:,1:r)'*xnoise);        
        Eproj(ii,k) = norm(xproj-x)/norm(x);

        % QR sensors      
        xqr = Psi(:,1:r)*(Psi(pivot,1:r)\xnoise(pivot));
        Eqr(ii,k) = norm(xqr-x)/norm(x);   
        
        % DEIM sensors
        xdeim = Psi(:,1:r)*(Psi(gamma,1:r)\xnoise(gamma));
        Edeim(ii,k) = norm(xdeim-x)/norm(x);   

        % Data matrix QR
        xdat = Psi(:,1:r)*(Psi(dpiv,1:r)\xnoise(dpiv));
        Edat(ii,k) = norm(xdat-x)/norm(x); 
        
        % Oversampled QR (SSPOR)
        xsspor = Psi(:,1:r)*(Psi(opiv,1:r)\xnoise(opiv));
        Esspor(ii,k) = norm(xsspor-x)/norm(x); 
    end
    
end


save([datpath,'ibpm_noise'],'var','Eproj','Edeim','Eqr','Edat','Esspor');

%% display convergence plot

close all;
aboxplot({Eproj,Esspor,Eqr,Edeim,},'labels',num2str(var'));
legend('POD','SSPOR','QDEIM','DEIM','Location','best');
set(gca,'yscale','log'); grid on
set(gca,'xTickLabelRotation',45)
set(gcf,'position',[380 320 2*540 400])

export_fig([figpath,'FIG_IBPM_NOISE.pdf'],'-transparent');

