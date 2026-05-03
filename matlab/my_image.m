function my_image(x1,x2,a,lambda,c,t0,Data,n,l0)
Z=zeros(1000,600);
parfor l=1:1000
     xi=l*0.0001;
    for m=1:600
        yi=m*0.0001;
        % Y(l,m,:)=my_stack(xi,yi,x1,x2,a,lambda,c,t0,data(:,:,:),20000,n,l0);
        % Z(l,m)=abs(Y(l,m,1));
        K=my_stack(xi,yi,x1,x2,a,lambda,c,t0,Data(:,:,:),n,l0);
        K_subset=abs(K(1:600));
        Z(l,m)=max(K_subset);
        % Z(l,m)=mean(K_subset);
    end
end
A=log10(Z+1);
A_norm=(A - min(A(:)))/(max(A(:)) - min(A(:)));
imagesc(A_norm');
colorbar;
axis image;