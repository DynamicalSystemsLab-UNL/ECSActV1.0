data_file = './full_solution/full_solution_s1/full_solution_s1_p0.h5';
dataT = h5read(data_file,'/scales/sim_time');
dataX_ = h5read(data_file,'/scales/x/1.0');
dataY_ = h5read(data_file,'/scales/y/1.0');
dataU = h5read(data_file,'/tasks/U');
dataV = h5read(data_file,'/tasks/V');
%dataQAA = h5read(data_file,'/tasks/QAA');
%dataQAB = h5read(data_file,'/tasks/QAB');
dataNx_ = h5read(data_file,'/tasks/nx');
dataNy_ = h5read(data_file,'/tasks/ny');
dataS = h5read(data_file,'/tasks/S');
dataUy = h5read(data_file,'/tasks/Uy');
dataVx = h5read(data_file,'/tasks/Vx');


dimX = length(dataX_); dimY = length(dataY_); NT = length(dataT);
NT_begin = 2; NT_end = NT;
width = 80; height = 20;
scale = 1; offset = width/dimX;

% framerate; dt = 1 takes every snapshot from the data file, dt = 2 
% takes every other, etc.
dt = 1; 

sx = 1; sy = 1; % Data reduction by factors of sx and sy; see below.

%---------------------------------------------------------------------%
% Scale the y coordinate as needed to mitigate distortions when 
% plotting the velocity and director fields.
% (If the x and y scales are too different, Matlab distorts certain 
% plot elements like arrows.
%---------------------------------------------------------------------%
for i = 1:dimY
    dataY_(i) = scale*dataY_(i);
end
%---------------------------------------------------------------------%


%---------------------------------------------------------------------%
% Turn 1d (x,y) coordinate arrays into 2D grids (required for plotting
% later
%---------------------------------------------------------------------%
dataX = zeros(dimY, dimX, NT);
dataY = zeros(dimY, dimX, NT);
for k = NT_begin:dt:NT_end
    for i = 1:dimY
        dataX(i,1:dimX, k) = dataX_;
    end
    for j = 1:dimX
    	dataY(1:dimY, j, k) = dataY_;
    end
end
%---------------------------------------------------------------------%

%---------------------------------------------------------------------%
% Normalize the director. For technical reasons, I could not get
% Dedalus to do this robustly.
%---------------------------------------------------------------------%
dataNx = zeros(dimY, dimX, NT);
dataNy = zeros(dimY, dimX, NT);
for i = 1:dimX
    for j = 1:dimY
        norm_temp = sqrt(dataNx_(j,i,:).*dataNx_(j,i,:) + dataNy_(j,i,:).*dataNy_(j,i,:));
        dataNx(j,i,:) = dataNx_(j,i,:)./norm_temp;
        dataNy(j,i,:) = abs(dataNy_(j,i,:))./norm_temp;
    end
end
%---------------------------------------------------------------------%

%---------------------------------------------------------------------%
% Vorticity and velocity magnitude
%---------------------------------------------------------------------%
dataVOR = zeros(dimY, dimX, NT);
dataUMAG = zeros(dimY, dimX, NT);
for i = 1:dimX
    for j = 1:dimY
        dataVOR(j,i,:) = dataVx(j,i,:) - dataUy(j,i,:);
        dataUMAG(j,i,:) = sqrt(dataV(j,i,:).*dataV(j,i,:) + dataU(j,i,:).*dataU(j,i,:));
    end
end
%---------------------------------------------------------------------%


%---------------------------------------------------------------------%
% Data reduction by factors of sx (or sx1) and sy. This is necessary
% if the quiver plot otherwise would produce more arrows than can
% be properly visualize
%---------------------------------------------------------------------%
X = dataX(1:sy:end,1:sx:end,:);
Y = dataY(1:sy:end,1:sx:end,:);
U = dataU(1:sy:end,1:sx:end,:);
V = dataV(1:sy:end,1:sx:end,:);
N_x = dataNx(1:sy:end,1:sx:end,:);
N_y = dataNy(1:sy:end,1:sx:end,:);
%---------------------------------------------------------------------%

%---------------------------------------------------------------------%
% The director has a head-tail symmetry, i.e. sign ambiguity.
% Therefore, to plot the director field, I construct two vector fields:
% one pointing in the +(nx, ny) direction and the other in the (-nx,
% ny) direction. Plotting both fields without arrow heads will then
% generate a field of rods representing the director.
%---------------------------------------------------------------------%
dimYr = ceil(dimY/sy); dimXr = ceil(dimX/sx);
Xr = zeros(2*dimYr, 2*dimXr, NT);
Yr = zeros(2*dimYr, 2*dimXr, NT);
N_xr = zeros(2*dimYr, 2*dimXr, NT);
N_yr = zeros(2*dimYr, 2*dimXr, NT);

Xr(1:dimYr,1:dimXr,:) = X; Xr(dimYr+1:2*dimYr,dimXr+1:2*dimXr,:) = X;
Yr(1:dimYr,1:dimXr,:) = Y; Yr(dimYr+1:2*dimYr,dimXr+1:2*dimXr,:) = Y;
N_xr(1:dimYr,1:dimXr,:) = N_x(1:dimYr,1:dimXr,:); N_xr(dimYr+1:2*dimYr,dimXr+1:2*dimXr,:) = -N_x(1:dimYr,1:dimXr,:);
N_yr(1:dimYr,1:dimXr,:) = N_y(1:dimYr,1:dimXr,:); N_yr(dimYr+1:2*dimYr,dimXr+1:2*dimXr,:) = -N_y(1:dimYr,1:dimXr,:);
%---------------------------------------------------------------------%

figure, set(gcf, 'Color','black')
set(gca, 'nextplot','replacechildren', 'Visible','off');
vidObj = VideoWriter('video.avi');
vidObj.Quality = 100;
vidObj.FrameRate = 20;
open(vidObj);
for k=NT_begin:dt:NT_end
    t = tiledlayout(2,1);
    nexttile
    contourf(dataX(:,:,k),dataY(:,:,k),dataS(:,:,k),100,'LineColor','none')
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    xlim([-0.5*width 0.5*width-offset]) 
    ylim([0 scale*height])
    daspect([1 1 1])
    set(gca,'FontSize',26)
    set(gca, 'XColor',[0 0 0])
    set(gca, 'YColor',[0 0 0])
    
    % Custom tickmarks are required because the y-coordinate was scaled by
    % factor 'scale'.
    xticks([-0.5*width -20 -10 0 10 20 0.5*width-offset])
    yticks([0*scale 5*scale 10*scale 15*scale 20*scale])
    ax = get(gca);
    ax.TickLabelInterpreter = 'tex';
    xticklabels({'\color[rgb]{0.9,0.9,0.9} -40','\color[rgb]{0.9,0.9,0.9} -20','\color[rgb]{0.9,0.9,0.9} -10','\color[rgb]{0.9,0.9,0.9} 0','\color[rgb]{0.9,0.9,0.9} 10','\color[rgb]{0.9,0.9,0.9} 20','\color[rgb]{0.9,0.9,0.9} 40'})
    yticklabels({'\color[rgb]{0.9,0.9,0.9} 0','\color[rgb]{0.9,0.9,0.9} 5','\color[rgb]{0.9,0.9,0.9} 10','\color[rgb]{0.9,0.9,0.9} 15','\color[rgb]{0.9,0.9,0.9} 20'})
    
    set(gca,'TickLength',[0.0025, 0.0025])
    cb = colorbar;
    cb.Color = [0.9,0.9,0.9];
    caxis([0.1 1])
    s_temp = "{\color{white}Nematic director field;        t = }" + sprintf('%f',round(1000*dataT(k))/1000);
    title(s_temp, 'Color','white','FontSize',26)
    hold on
    quiver(Xr(:,:,k),Yr(:,:,k),N_xr(:,:,k),N_yr(:,:,k),0.7,'ShowArrowHead',0,'LineWidth',0.9,'Color',[0.1,0.1,0.1])
    hold off
    
    nexttile
    contourf(dataX(:,:,k),dataY(:,:,k),dataVOR(:,:,k),100,'LineColor','none')
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    xlim([-0.5*width 0.5*width-offset]) 
    ylim([0 scale*height])
    daspect([1 1 1])
    set(gca,'FontSize',26)
    set(gca, 'XColor',[0 0 0])
    set(gca, 'YColor',[0 0 0])

    % Custom tickmarks are required because the y-coordinate was scaled by
    % factor 'scale'.
    xticks([-0.5*width -20 -10 0 10 20 0.5*width-offset])
    yticks([0*scale 5*scale 10*scale 15*scale 20*scale])
    ax = get(gca);
    ax.TickLabelInterpreter = 'tex';
    xticklabels({'\color[rgb]{0.9,0.9,0.9} -40','\color[rgb]{0.9,0.9,0.9} -20','\color[rgb]{0.9,0.9,0.9} -10','\color[rgb]{0.9,0.9,0.9} 0','\color[rgb]{0.9,0.9,0.9} 10','\color[rgb]{0.9,0.9,0.9} 20','\color[rgb]{0.9,0.9,0.9} 40'})
    yticklabels({'\color[rgb]{0.9,0.9,0.9} 0','\color[rgb]{0.9,0.9,0.9} 5','\color[rgb]{0.9,0.9,0.9} 10','\color[rgb]{0.9,0.9,0.9} 15','\color[rgb]{0.9,0.9,0.9} 20'})
    
    set(gca,'TickLength',[0.0025, 0.0025])
    cb = colorbar;
    cb.Color = [0.9,0.9,0.9];
    %cb2 = colorbar('south')
    %caxis([0 0.13])
    set(gca,'FontSize',26)
    title('Vorticity', 'Color','white')
    hold on
    quiver(X(:,:,k),Y(:,:,k),U(:,:,k),V(:,:,k),'LineWidth',0.75,'Color','black')
    hold off

    writeVideo(vidObj, getframe(gcf));
end
close(gcf)
close(vidObj);
