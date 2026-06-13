%% error v batch size IN1K [bs 64 needs updating, drop 128k?]
l={'64' '128' '256' '512' '1k' '2k' '4k' '8k' '16k' '32k' '64k'};
y=[23.48 23.49 23.60 23.48 23.53 23.49 23.56 23.74 24.79 27.55 33.96];
e=[0.00 0.12 0.12 0.09 0.08 0.11 0.12 0.09 0.27 0.28 0.80]*2;
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 600 300]);
hold on; col=[.2 .65 .9]; fc='MarkerFaceColor'; n=length(y); x=1:n;
yp=[y-e,fliplr(y+e)]; xp=[x,fliplr(x)];
patch(xp,yp,1,'facecolor',col,'edgecolor','none','facealpha',.3);
plot(x,y,'-o','color',col,fc,col,'LineWidth',2,'MarkerSize',4); grid on;
set(gca,'FontSize',16,'Xtick',x,'XTickLabel',l); axis([1 n 20 40]);
xlabel('mini-batch size'); ylabel('ImageNet top-1 validation error');
savepdf('batchsize');

%% error v batch size IN5K
l={'256' '512' '1k' '2k' '4k' '8k' '16k' '32k' '64k'};
y=[25.83 25.80 25.71 25.71 25.90 26.09 27.01 30.46 37.48];
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 600 300]);
hold on; col=[.2 .65 .9]; fc='MarkerFaceColor'; n=length(y); x=1:n;
plot(x,y,'-o','color',col,fc,col,'LineWidth',3,'MarkerSize',4); grid on;
set(gca,'FontSize',16,'Xtick',x,'XTickLabel',l); axis([1 n 20 40]);
xlabel('mini-batch size'); ylabel('ImageNet top-1 validation error');
savepdf('batchsizeIN5k');

%% throughput
x = [2.^(0:5) 44]*8;
t = [0.23 0.241 0.247 0.246 0.25 0.257 0.257];
yr = 32*x./t; yo = 32*x./t(1);
yt = {'1k' '2k' '4k' '8k' '16k' '32k'};
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 600 280]);
c1=[.37 .3 .8]; c2=[.97 .46 .3]; fc={'MarkerSize',4,'MarkerFaceColor'};
h(1)=loglog(x,yo,'-o','LineWidth',2,'color',c1,fc{:},c1); hold on;
h(2)=loglog(x,yr,'-o','LineWidth',2,'color',c2,fc{:},c2);
legend(h,{'ideal','actual'},'Location','nw','FontSize',16);
set(gca,'FontSize',16,'XTick',x,'YTick',2.^(0:5)*1024,'YTickLabel',yt);
xlabel('# GPUs'); ylabel('images / second'); axis tight; grid on;
set(gca,'XMinorTick','off','XMinorGrid','off');
set(gca,'YMinorTick','off','YMinorGrid','off');
savepdf('throughput');

%% color picker >:(
figure(1); clf; x=1:100; y1=x; y2=x*1.2;
cs=(1:100)'; cs=max(.3,mod([.306*cs .475*cs .165*cs],1));
for i=1:25
  subplot(5,5,i);
  c1=cs(randi([1 100],1,1),:); c2=cs(randi([1 100],1,1),:);
  L={sprintf('%.3f ',c1) sprintf('%.3f ',c2)};
  h(1)=plot(x,y1,'-','LineWidth',4,'color',c1); hold on;
  h(2)=plot(x,y2,'-','LineWidth',4,'color',c2);
  legend(h,L,'Location','nw','FontSize',8);
  set(gca,'XTick',[],'YTick',[]); axis tight; grid off;
end

%% time
x = [2.^(0:5) 44]*8*32;
ti = [0.23 0.241 0.247 0.246 0.25 0.257 0.257];
te = round(1281167./x.*ti)/60;
xt = {'256' '512' '1k' '2k' '4k' '8k' '11k'};
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 600 300]);
c1=[.37 .3 .8]; c2=[.97 .46 .3]; fc='MarkerFaceColor';
yyaxis left; semilogx(x,ti,'-o','LineWidth',2,'color',c1,fc,c1);
set(gca,'yColor',c1,'YTick',.2:.02:.3,'YMinorTick','off');
ylabel('time per iteration (secs)'); axis([x(1) x(end) .2 .3])
yyaxis right; loglog(x,te,'-o','LineWidth',2,'color',c2,fc,c2);
set(gca,'yColor',c2,'YTick',2.^(-1:4),'YMinorTick','off');
ylabel('time per epoch (mins)'); axis tight;
set(gca,'FontSize',16,'Xtick',x,'XMinorTick','off','XTickLabel',xt);
xlabel('mini-batch size'); grid off; savepdf('time');

%% training curve  plots
fmt='\\it{kn}\\rm=%3s, \\eta=%4.1f, %.2f%%\\pm%.2f';
switch 1
  case 1 % diffent types of warmup
    is = [18330972 18332795 18332868 18331260];
    es = [23.60 24.84 25.88 23.74; 0.12 0.37 0.56 0.09];
    ns = {'1-node','none','constant','gradual'};
    name='warmup'; n=length(is); L=cell(1,n);
    for i=1:n, L{i}=sprintf(fmt,'8k',3.2,es(:,i)); end
    L{1}=sprintf(fmt,'256',.1,es(:,1));
  case 2 % different mini-batch sizes
    is = [18330972 18332373 18331033 18331074 18331104 ...
      18331213 18331260 18331310 18331348 18331462];
    es = [23.60 23.49 23.48 23.53 23.49 23.56 23.74 24.79 27.55 33.96;
      0.12 0.12 0.09 0.08 0.11 0.12 0.09 0.27 0.28 0.80 ];
    ns = {'256' '128' '512' '1k' '2k' '4k' '8k' '16k' '32k' '64k'};
    name='mb'; n=length(is); L=cell(1,n);
    for i=1:n, L{i}=sprintf(fmt,ns{i},.1*2^(i-2),es(:,i)); end
    L{2}(23:26)='0.05';
  case 3 % different lr
    is = [18330972 18772756 18773014];
    es = [23.60 23.92 23.68; 0.12 0.1 0.09];
    ns = {'10','05','20'}; lr=[.1 .05 .2];
    name='lr'; n=length(is); L=cell(1,n);
    for i=1:n, L{i}=sprintf(fmt,'256',lr(i),es(:,i)); end
    L{2}(23:26)='0.05';
  case 4 % different lr scaling rules
    is = [18330972 18416193 18332283];
    es = [23.60 41.78 26.28; 0.12 0.1 0.03];
    ns = {'1-node','none','sqrt'}; lr=[.1 .1 .566];
    name='scaling'; n=length(is); L=cell(1,n);
    for i=1:n, L{i}=sprintf(fmt,'8k',lr(i),es(:,i)); end
    L{1}=sprintf(fmt,'256',.1,es(:,1));
end
[x,y]=loadLogs(is); ylbl='training error %'; fs='FontSize';
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 400 280]);
for i=1:n-1, clf;
  h(1)=plot(x{1+0},y{1+0},'Color',[.97 .58 .12]); hold on;
  h(2)=plot(x{i+1},y{i+1},'Color',[.16 .67 .89]); grid on;
  set(h,'LineWidth',2); if(i>=n-3), xlabel('epochs'); end
  if(mod(i,3)==1), ylabel(ylbl); else set(gca,'YTickLabel',[]); end
  h=legend(h,{L{1},L{i+1}}); axis([0 90 20 100]);
  set(gca,fs,12); set(h,'fontname','fixedwidth',fs,13)
  savepdf(sprintf('distr-%s-%s',name,ns{i+1}));
end

%% train+val for single setting
ids = [18330972 10013];
figure(1); clf; set(1,'Units','Pixels','Position',[50 800 400 240]);
[x,es0,es1]=loadLogs(ids);
h(1)=plot(x{1},es0{1},'-','Color',[.97 .58 .12]); hold on;
h(2)=plot(x{1},es1{1},':','Color',[.97 .58 .12]);
h(3)=plot(x{2},es0{2},'-','Color',[.16 .67 .89]);
h(4)=plot(x{2},es1{2},':','Color',[.16 .67 .89]);
set(h,'LineWidth',2); xlabel('epochs'); ylabel('error %');
nms={'256, \eta=0.1 [train]', '256, \eta=0.1 [val]',...
  '8k,  \eta=3.2 [train]', '8k,  \eta=3.2 [val]'};
for i=1:length(nms), nms{i}=['\it{kn}\rm=' nms{i}]; end
h=legend(h,nms); axis([0 90 20 100]); set(gca,'FontSize',12);
set(h,'fontname','fixedwidth','FontSize',14); grid on;
savepdf('train+val');
