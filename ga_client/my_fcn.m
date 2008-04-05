function out=my_fcn(X)
global A

CD=A(end-1:end,:);
a = A(1:end-2, :);

[m,n]=size(a);

a=reshape(a,1,m*n);
X = X + CD(2, :);
Xext=X'*X;
Xext=reshape(Xext,1,m*n);

out=a*Xext'+CD(1,:)*X';