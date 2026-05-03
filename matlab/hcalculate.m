function h=hcalculate(x,y,x1,x2,a,lambda,m,c,t0)%x,y为聚焦点位置，x1,x2分别为发射和接收探头位置，lambda为波长，a为探头半径,m为数据，c为波速，t0为采样间隔
p1=pcalculate(x,y,x1,lambda,a);
p2=pcalculate(x,y,x2,lambda,a);
d_x1=norm([x1,0]-[x,y]);
d_x2=norm([x2,0]-[x,y]);
k=(p1*p2*a^2)/(d_x1*d_x2);
x0=round(((d_x1+d_x2)/c)/t0);
%fprintf('%f\n',x0);
m=reshape(m,1,[]);
len_m = length(m);
h = zeros(1, len_m);
%k=1;
if x0 < len_m
    h(1:len_m-x0)=m(x0 + 1:end)/k;
end