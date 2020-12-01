%% PAM application on segmented data
% Written on 29.03.16 by Sinan Acikgoz (msa44@cam.ac.uk)
% L4 input files provided. This corresponds to section L1 investigated in
% the paper. 


%% Get the data for sections you are comparing
clear
cloud1='12_12_14_L4';% Undeformed cloud name initials 
cloud2='13_11_23_L4';% Deformed cloud name initials 
gridstep=0.001; 
toldisp=0.01; % Defines the range of values of Y where the maximum/minimum Y edges of the rotated segment will be found. Not critical. 
loop=28; % Number of segments that are investigated 
tmat=zeros(loop,7);
ang=-29.005;% degrees of rotation of the arch Northing, Easting and Elevation coordinates on London grid
dis=zeros(2*loop,4);
crcoord=zeros(2*loop,3);
inix=-0.0085; % X Mean Displacement of control point obtained from Table 3 
stdx=0.0006;% Std Dev of control point 
%% Parametric values
paraid=1;  % 0 means normal PAM, % 1 means IPAM
loop2=200; 
factormmt=0.25; %factormmt*loop2/4 gives the range of X&Y movement in IPAM in mm. 
% factormmt is the step size in between applied movements for alignments
worstrej=0.1; % rejection of worst points in ICP. Potential noisy points. 
% factorz =[0.9,1.1]; 
limx = 0.001; %error limit for displacement continuity at the edges of segments in X
limy = 0.005; % in Y 
limz= 0.00025; % in X 
absy = 0.01;
relx = 0.0025;

for i=1:loop 
    % Prepare variables 
    close all  
    clearvars -except tmat dis dataspr cloud1 cloud2 i loop...
        loop2 toldisp ang datam dataf crcoord inix stdx ...
        factormmt worstrej limz limx limy absy paraid relx
    dispev=zeros(loop2*2,3);
    % Import segment data to be matched
    datam=importfile([cloud1,'_V',num2str(i),'.txt'],2,inf); % raw data for cloud1 segment
    dataf=importfile([cloud2,'_V',num2str(i),'.txt'],2,inf);  % raw data for cloud2 segment
    dataml=zeros(size(datam)); datamdl=zeros(size(datam)); datafl=zeros(size(dataf));
    % Rotate data to local axis and identify critical points
    rmat2=SpinCalc('EA321toDCM',[ang,0,0],0.0001,1); 
    rmat3=SpinCalc('EA321toDCM',[-ang,0,0],0.0001,1);
    for k=1:length(datam)
        dataml(k,:)=(rmat2*datam(k,:).').'; % Data rotated so that the horizontal X is perpendicular to arch axis
    end
    index=(dataml(:,2)>mean(dataml(:,2))-toldisp &...
        dataml(:,2)<mean(dataml(:,2))+toldisp); 
    % Raw cloud segments are 0.35m thick, presented data is from thinner
    % central strips, defined by 2*toldisp
    cloudsl1=dataml(index,:); % Investigated central strip of segment
    datamin=rmat3*(cloudsl1(cloudsl1(:,1)==min(cloudsl1(:,1)),:)).'; % edge of segment with min values of X, in unrotated coord
    datamax=rmat3*(cloudsl1(cloudsl1(:,1)==max(cloudsl1(:,1)),:)).'; % edge of segment with max values of X, in unrotated coord
    datamin=datamin(:,1); datamax=datamax(:,1);
    if i==1
        dataspr=datamin; % Definition of springing point of segment in unrotated coord
    end
    crcoord((2*i-1):2*i,:)=[(rmat2*(datamin-dataspr)).';...
        (rmat2*(datamax-dataspr)).']; % Definition of the rotated edge coordinates
%     figure
%     plot3(datam(:,1),datam(:,2),datam(:,3),'k.')
%     hold on
%     plot3(datamin(1),datamin(2),datamin(3),'rs','MarkerSize',15)
% %     plot3(datamean(1),datamean(2),datamean(3),'rs','MarkerSize',15)
%     plot3(datamax(1),datamax(2),datamax(3),'rs','MarkerSize',15)
    %    Do the ICP fit 
    if i==1
        iniali=mean(dataf)-mean(datam); % initial alignment puts the CoG of data together. Here it's defined 
        iniali(1)=inix+limx;
    else
        iniali=dis(2*(i-1),2:4)+[limx 0 0];
    end
    for t=1:loop2  
        clear rmat tvec error 
        mmt=sign((-1)^(t+1))*ceil(t/4)/1e3*factormmt; 
        % This is the iterative element which changes the initial alignment in X and Y. 
        % factormmt 
        if paraid==0
            inialim=[0,0,0];
        elseif mod(t,4)==1 || mod(t,4)==2 
            inialim=iniali+(rmat3*[mmt 0 0].').'; % Initial alignment + translation for IPAM in X
        else
            inialim=iniali+(rmat3*[0 mmt 0].').'; % Initial alignment + translation for IPAM in Y
        end
        minl=min(length(dataf),length(datam));
        dataficp=dataf(random('unid',length(dataf),ceil(minl),1),:);
        datamicp=datam(random('unid',length(datam),ceil(minl),1),:);
        tra=ones(ceil(minl),3); 
        tra(:,1)=inialim(1)*tra(:,1); 
        tra(:,2)=inialim(2)*tra(:,2);
        tra(:,3)=inialim(3)*tra(:,3); % translation matrix defined according to inialim
        [rmat,tvec,error] = icp(dataficp.',(datamicp+tra).',4,'Matching','kDtree',...
            'WorstRejection',worstrej);
        % rmat is the rotation that is applied to the translated datam
        % cloud. tvec is its translation. error is the error of fitting 
        % in RMS for corresponding points        
        dispev(2*t-1,:)=(rmat2*((rmat*(datamin+inialim.'))+tvec-datamin)).';
        dispev(2*t,:)=(rmat2*((rmat*(datamax+inialim.'))+tvec-datamax)).';
        % These are the deflections of the edge points, which are rotated 
        % in the barrel axis.         
        if i==1 && (dispev(2*t-1,1)<=inix+stdx   &&... 
                dispev(2*t-1,1)>=inix-stdx   &&...
                abs(dispev(2*t-1,2))<absy) ;
            tmat(i,:)=[SpinCalc('DCMtoEA321',rmat2*rmat,0.0001,1),tvec.', error(end)];
            break
        elseif  i~=1 && dispev(2*t-1,3)<=(dis(2*i-2,4))+limz &&...
                dispev(2*t-1,3)>=dis(2*i-2,4)-limz &&...
                dispev(2*t-1,1)<=dis(2*i-2,2)+limx   &&...
                dispev(2*t-1,1)>=dis(2*i-2,2)-limx  &&...
                dispev(2*t-1,2)<=dis(2*i-2,3)+limy  &&...
                dispev(2*t-1,2)>=dis(2*i-2,3)-limy &&...
                abs(dispev(2*t-1,2))<absy &&...
                abs(dispev(2*t,2))<absy;
            tmat(i,:)=[SpinCalc('DCMtoEA321',rmat2*rmat,0.0001,1),tvec.', error(end)];
            break  
        elseif t==loop2; 
            if i==1
                bestfit=zeros(size(dispev));
                bestfit(:,1)=dispev(:,1);
                bestfit(:,2)=dispev(:,2); 
                bestfit(abs(dispev(:,2))>absy,1)=1;
                bestfit(:,3)=dispev(:,3);
                bestfit(:,4)=sqrt((bestfit(:,1)-inix).^2);              
            else
                bestfit=zeros(size(dispev));
                bestfit(:,1)=dispev(:,1)-dis(2*i-2,2)*ones(length(dispev),1);
                bestfit(:,2)=dispev(:,2)-dis(2*i-2,3)*ones(length(dispev),1);                
                bestfit(:,3)=dispev(:,3)-dis(2*i-2,4)*ones(length(dispev),1);
                bestfit(abs(bestfit(:,1))>relx,3)=1; 
                bestfit(abs(dispev(:,2))>absy,2)=1;
                bestfit(abs(dispev(:,2))>absy,3)=1;
                bestfit(2*(find(bestfit(2:2:end,2)==1))-1,3)=1;
                %bestfit(:,4)=bestfit(:,1).^2+bestfit(:,2).^2+bestfit(:,3).^2;
                bestfit(:,4)=bestfit(:,1).^2+bestfit(:,3).^2;
            end
            t=find(bestfit(1:2:end,4)== min(bestfit(1:2:end,4)),1);
            mmt=sign((-1)^(t+1))*ceil(t/4)/1e3*factormmt;
            if mod(t,4)==1 || mod(t,4)==2
                inialim=iniali+(rmat3*[mmt 0 0].').'; % Initial alignment + translation for IPAM in X
            else
                inialim=iniali+(rmat3*[0 mmt 0].').'; % Initial alignment + translation for IPAM in Y
            end
            minl=min(length(dataf),length(datam));
            dataficp=dataf(random('unid',length(dataf),ceil(minl),1),:);
            datamicp=datam(random('unid',length(datam),ceil(minl),1),:);
            tra=ones(ceil(minl),3);
            tra(:,1)=inialim(1)*tra(:,1);
            tra(:,2)=inialim(2)*tra(:,2);
            tra(:,3)=inialim(3)*tra(:,3); % translation matrix defined according to inialim
            [rmat,tvec,error] = icp(dataficp.',(datamicp+tra).',4,'Matching','kDtree',...
                'WorstRejection',worstrej);
%         dispev(2*t-1,:)=(rmat2*((rmat*(datamin+inialim.'))+tvec-datamin)).';
%         dispev(2*t,:)=(rmat2*((rmat*(datamax+inialim.'))+tvec-datamax)).';
            tmat(i,:)=[SpinCalc('DCMtoEA321',rmat2*rmat,0.0001,1),tvec.', error(end)];
        end
    end
    tra=ones(size(datam));
    tra(:,1)=inialim(1)*tra(:,1);
    tra(:,2)=inialim(2)*tra(:,2);
    tra(:,3)=inialim(3)*tra(:,3);
    dis((2*i-1):2*i,:)= [norm(datamin(1:2)-dataspr(1:2)), dispev(2*t-1,:);...
        norm(datamax(1:2)-dataspr(1:2)), dispev(2*t,:)];
    figure 
    for k=1:length(datam)
        datamdl(k,:)=(rmat*(datam(k,:)+tra(k,:)).'+tvec).';
    end
     plot3(datam(:,1),datam(:,2),datam(:,3),'.','Color',[0.7 0.7 0.7])
     hold on 
     plot3(datam(:,1)+tra(:,1),datam(:,2)+tra(:,2),datam(:,3)+tra(:,3),'k.')
     plot3(dataf(:,1),dataf(:,2),dataf(:,3),'g.')
     figure
     plot3(datamdl(:,1),datamdl(:,2),datamdl(:,3),'k.')
     hold on
     plot3(dataf(:,1),dataf(:,2),dataf(:,3),'g.')
end      
   
tmat(tmat(:,1)>180,1)=tmat(tmat(:,1)>180,1)-360;
tmat(tmat(:,2)>180,2)=tmat(tmat(:,2)>180,2)-360;
tmat(tmat(:,3)>180,3)=tmat(tmat(:,3)>180,3)-360;
close all
% Plot critical results 
% figure  
% subplot(2,2,1)
% plot(dis(:,1),dis(:,2)*1e3)
% title('Displacements along X (mm)') 
% subplot(2,2,2)
% plot(dis(:,1),dis(:,4)*1e3)
% title('Displacements along Z (mm)') 
% subplot(2,2,3)
% plot(dis(:,1),dis(:,3)*1e3)
% title('Displacements along Y (mm)') 
% xlabel('Distance along X from western pier wall') 
% subplot(2,2,4)
% plot(crcoord(:,1),crcoord(:,3),crcoord(:,1)+1e2*dis(:,2),crcoord(:,3)+1e2*dis(:,4))
% title('Undeformed and deformed geometries, def scale=100')
% savefig([cloud1,'vs',cloud2,'displacements','_para',num2str(paraid)]);

% figure 
% subplot(2,2,1) 
% plot(dis(1:2:end,1),tmat(:,1))
% title('Rotations along Z (degrees)') 
% xlabel('Distance along X from western pier wall')
% subplot(2,2,2)
% plot(dis(1:2:end,1),tmat(:,2))
% title('Rotations along Y (degrees)') 
% xlabel('Distance along X from western pier wall')
% subplot(2,2,3) 
% plot(dis(1:2:end,1),tmat(:,3))
% title('Rotations along X (degrees)') 
% xlabel('Distance along X from western pier wall')
% subplot(2,2,4)
% plot(dis(1:2:end,1),tmat(:,7))
% title('Fitting error with each section') 
% ylabel('RMS error (mm)')
% xlabel('Distance along Xfrom western pier wall')
% % savefig([cloud1,'vs',cloud2,'rotations','_paraV',num2str(paraid)]);
% 
figure
hold on
plot(crcoord(:,1),crcoord(:,3),crcoord(:,1)+1e2*dis(:,2),crcoord(:,3)+1e2*dis(:,4))
q=quiver(crcoord(:,1),crcoord(:,3),dis(:,2)*1e2,dis(:,4)*1e2,'r.', 'MarkerSize',6);
q.AutoScale='off';
q.ShowArrowHead='on';
q.MaxHeadSize=0.1;
title('Displacement vectors and deflected shape (deformations scaled x100)') 
xlabel('Distance along X from western pier wall')
savefig([cloud1,'vs',cloud2,'PAPERdeflectedshape','_paraV',num2str(paraid)]);


% Smoothen crcoord
dissm=zeros(length(dis(:,1))/2+1,4);
dissm(1,:)=dis(1,:);
dissm(end,:)=dis(end,:);
for i=1:length(dis(:,1))/2-1
    dissm(i+1,:)=(dis(2*i,:)+dis(2*i+1,:))./2;
end

figure 
subplot(2,2,1)
plot(dis(:,1),dis(:,4)*1e3)
axis([-1, 10.6, -45, 5])
ylabel('\Delta\itZ\rm, vertical disp (mm)') 
grid minor
subplot(2,2,2)
plot(dis(:,1),dis(:,2)*1e3)
axis([-1, 10.6, -20, 20])
ylabel('\Delta\itX\rm, lateral disp (mm)') 
grid minor
subplot(2,2,3)
plot(dis(1:2:end,1),tmat(:,2))
axis([-1, 10.6, -0.5, 1])
ylabel('\Delta\it\theta_Y\rm, in-plane rot (degrees)') 
xlabel('X, dist along barrel axis (m)')
grid minor
subplot(2,2,4)
plot(crcoord([1:2:end,end],1),crcoord([1:2:end,end],3),crcoord([1:2:end,end],1)+1e2*dissm(:,2),crcoord([1:2:end,end],3)+1e2*dissm(:,4))
axis([-1 10.6 -5 2.5])
ylabel('Deflected shape')
grid minor
legend('Undeformed', 'Deformed (scale=100)')
xlabel('X (m)')
savefig([cloud1,'vs',cloud2,'PAPERdisps','_paraV',num2str(paraid)]);

% 
%     dis((2*i-1):2*i,:)= [norm(datamin(1:2)-dataspr(1:2)),(rmat2*(rmat*(datamin+inialim.')+tvec-datamin)).';...
%         norm(datamean(1:2)-dataspr(1:2)),(rmat2*(rmat*(datamean+inialim.')+tvec-datamean)).';
%         norm(datamax(1:2)-dataspr(1:2)),(rmat2*(rmat*(datamax+inialim.')+tvec-datamax)).'];
%      for k=1:length(datam)
%         dataml(k,:)=(rmat2*datam(k,:).').';
%         datamdl(k,:)=(rmat2*(rmat*(datam(k,:)+tra(k,:)).'+tvec)).';
%     end
%     %might delete later- section below
%     for n=1:length(dataf)
%         datafl(n,:)=(rmat2*dataf(n,:).').'; 

    %datamean=rmat3*(cloudsl1(cloudsl1(:,1)==median(cloudsl1(:,1)),:)).';

%     if isempty(datamean)==1
%         datamean=rmat3*(cloudsl1(cloudsl1(1:end-1,1)==median(cloudsl1(1:end-1,1)),:)).'; 
%     else






