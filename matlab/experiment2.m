CoreNum=8; %设定机器CPU核心数量
if isempty(gcp('nocreate')) %如果并行未开启
    parpool(CoreNum);
end
files={'00.mat','01.mat','02.mat','03.mat';
    '10.mat','11.mat','12.mat','13.mat';
    '20.mat','21.mat','22.mat','23.mat';
    '30.mat','31.mat','32.mat','33.mat';};
data=my_read(files,400000,4);
% fprintf('%d\n',length(data));
l0=0.002;%移动间隔
c=6270;%声速
t0=5e-10;%采样时间
a=5e-3;%探头底面半径
lambda=1.254e-3;
d=1e-2;%探头外直径 用于算位置
x1=d/2;%发射探头中心
x2=0.0448+d/2;%接收探头中心
n=4;%等效探头数量
my_image(x1,x2,a,lambda,c,t0,data(:,:,:),n,l0);
my_stack(0.02,0.02,x1,x2,5e-3,lambda,c,t0,data(:,:,:),n,l0);
