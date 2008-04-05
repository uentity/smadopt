function [ best_ind, best_c, best_fcn]=my_kmeans2(X,Ncl)
%X - data matrix [nxp], in which row -is array of signs, 
%    n -number of rows - number of points 
%    in p-dimensional space
%Ncl - number of clusters
%

%global X;


Ncyc=60;

%type of closes
%1- euclid norm
%2- cos norm  alf=1-abs(cos)
type_cl=1;

%size vectors:
%p - vector length
[Nx,p]=size(X);
%0. end

best_fcn=[];
ccyc=1;
while ccyc<60+1%Nx+1
    %1. first cluser centers
    newc=X(ccyc,:);
    [val,ind,sumerr]=ind_cl_gr(newc,X,1,2);
    [sorterr inderr]=sort(sumerr);
    
    destep=(length(sumerr)-1)/Ncl;
    indx=floor([1:destep:length(sumerr)]);
    newc=X(inderr(indx),:);
    %1. end


    icyc=1;
    while icyc<Ncyc+1
        c=newc;
        [val,ind,sumerr]=ind_cl_gr(c,X,Ncl,type_cl);
        newc=new_cl(ind,X,Ncl);
        goal_fcn=sum(val);
        
        if length(best_fcn)==0
            best_fcn=goal_fcn;
            best_c=c;
            best_ind=ind;
        end
        if goal_fcn<best_fcn
            best_fcn=goal_fcn;
            best_c=c;
            best_ind=ind;
        end
        icyc=icyc+1;
    end
    %2. end
    ccyc=ccyc+1;
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%subfunctions

%s1.---------------------------------------------------------
function [val,ind,sumerr]=ind_cl_gr(c,X,Ncl,type_cl)
%fcn find distanecs for each point near nearest cluster 
%c - clusters [Nclxp] in p-dim
%X - Nx points in p dim
%Ncl - number of clusters
%type_cl - type of clusterization
% degrees of membership are defined

[Nx,p]=size(X);
i=1;
while i<Ncl+1
%1 - euclid norm
%2 - cos norm
c=c;
   if type_cl==1
       c(i,:);
       cbuf=repmat(c(i,:),Nx,1);
       errbuf=cbuf-X;
       sumerr(:,i)=(sum((errbuf.*errbuf)')');
   elseif type_cl==2
       cbuf=c(i,:);
       cosm=cbuf*X';
       j=1;
       while j<Nx+1
           Xj=X(j,:);
           normX(1,j)=norm(Xj,2);
           j=j+1;
       end
      cosm=cosm./normX/norm(cbuf);
       sumerr(:,i)=1-cosm;
    end
    i=i+1;
end
[val ind]=min(sumerr');
%s1. end------------------------------------------------


%s2. new clusters ----------------------------------------
function newc=new_cl(ind,X,Ncl)
i=1;
while i<Ncl+1
   bufX=X(find(ind==i),:);
   if length(bufX)==0
       newc(i,:)=X(1,:)*0;
   else
       if size(bufX,1)==1
           newc(i,:)=bufX;
       else
           newc(i,:)=median(bufX);
       end
   end
   i=i+1;
end


%s2, end --------------------------------------------------

