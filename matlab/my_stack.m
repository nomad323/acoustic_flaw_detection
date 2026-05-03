function result=my_stack(x,y,x1,x2,a,lambda,c,t0,Data,n,d)%x,y为聚焦点，x1,x2分别为探头起始中心位置，a为半径，lambda为波长，c为波速，t0为采样间隔,t为每一次扫描大小，data为数据,n为探头个数(移动数+1),d为移动距离
len=size(Data,3);
%fprintf('%d\n',len);
%fprintf('%f\n',Data(1,1,10000));
result=zeros(1,len);
x1_array = x1+(0:n-1)*d;
x2_array = x2+(0:n-1)*d;
for i=1:n
    xi=x1_array(i);
    for j=1:n
        xj=x2_array(j);
        s=hcalculate(x,y,xi,xj,a,lambda,Data(i,j,:),c,t0);
        %fprintf('%f\n',s(100));
        result=result+s(1,:);
    end
end
fs = 1e9;
fc = 2.5e6; % 中心频率 2.5 MHz
bw = 0.5e6; % 带宽 0.5 MHz
Wn = [fc - bw/2, fc + bw/2] / (fs / 2); % 归一化频率
% 设计带通滤波器
[b, a] = butter(4, Wn, 'bandpass');
% 应用滤波器
result=filter(b, a,result);
% x=1:len;
% result=abs(result);
% plot(x,log(result+1));