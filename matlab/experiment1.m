CoreNum=8; %设定机器CPU核心数量
if isempty(gcp('nocreate')) %如果并行未开启
    parpool(CoreNum);
end
files={'27.mat','37.mat';
    '28.mat','38.mat'};
data=my_read(files,20000,2);
h0=0.02;%深度
l0=0.01;%移动间隔
l1=0.02;%横向坐标
c=6270;%声速
t0=10^-9;%采样时间
a=1.0*10^-2;
lambda=2.368*10^-3;
d1=3.1*10^-2;%接收探头外直径
d2=3.2*10^-2;%发射探头外直径
% Y=zeros(1000,600,20000);
x1=0.07 + d2/2;%发射探头中心  
x2=0.02 + d1/2;%接收探头中心
n=2;%等效探头数量
delta=0.0001;%扫描间隔
my_image(x1,x2,a,lambda,c,t0,data(:,:,:),n,l0,delta);
