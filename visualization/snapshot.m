
    data_file = "./full_solution/full_solution_s1/full_solution_s1_p0.h5";
    dataT = h5read(data_file,'/scales/sim_time');
    dataX_ = h5read(data_file,'/scales/x/1.0');
    dataY_ = h5read(data_file,'/scales/y/1.0');
    %dataQAA_ = h5read(data_file,'/tasks/QAA');
    %dataQAB_ = h5read(data_file,'/tasks/QAB');
    dataNx_ = h5read(data_file,'/tasks/nx');
    dataNy_ = h5read(data_file,'/tasks/ny');
    dataS_ = h5read(data_file,'/tasks/S');
    dataU_ = h5read(data_file,'/tasks/U');
    dataV_ = h5read(data_file,'/tasks/V');
    dataUy_ = h5read(data_file,'/tasks/Uy');
    dataVx_ = h5read(data_file,'/tasks/Vx');

    dimX = length(dataX_); dimY = length(dataY_); NT = length(dataT);
    k = NT; % write index of snapshot
    
    width = 80; height = 20;
    scale = 1.0; offset = width/dimX;

    sx = 1; sy = 1; % Data reduction by factors of sx and sy; see below.
    sx1 = 1;
    %---------------------------------------------------------------------%
    
    % Select snapshot
    %dataQAA = dataQAA_(:,:,k);
    %dataQAB = dataQAB_(:,:,k);
    dataS = dataS_(:,:,k);
    dataU = dataU_(:,:,k);
    dataV = dataV_(:,:,k);
    dataUy = dataUy_(:,:,k);
    dataVx = dataVx_(:,:,k);
    
    % Scale the y coordinate as needed to mitigate distortions when 
    % plotting the velocity and director fields.
    % (If the x and y scales are too different, Matlab distorts certain 
    % plot elements like arrows.
    for i = 1:dimY
        dataY_(i) = scale*dataY_(i);
    end

    % Turn 1d (x,y) coordinate arrays into 2D grids (required for plotting
    % later
    dataX = zeros(dimY, dimX);
    dataY = zeros(dimY, dimX);
    for i = 1:dimY
        dataX(i,1:dimX) = dataX_;
    end
    for j = 1:dimX
        dataY(1:dimY, j) = dataY_;
    end

    % Normalize the director. For technical reasons, I could not get
    % Dedalus to do this robustly.
    dataNx = zeros(dimY, dimX);
    dataNy = zeros(dimY, dimX);
    for i = 1:dimX
        for j = 1:dimY
            norm_temp = sqrt(dataNx_(j, i)*dataNx_(j, i) + dataNy_(j, i)*dataNy_(j, i));
            dataNx(j, i) = dataNx_(j, i)/norm_temp;
            dataNy(j, i) = abs(dataNy_(j, i))/norm_temp;
        end
    end
    %---------------------------------------------------------------------%

    % Vorticity and velocity magnitude
    dataVOR = zeros(dimY, dimX);
    dataUMAG = zeros(dimY, dimX);
    for i = 1:dimX
        for j = 1:dimY
            dataVOR(j, i) = dataVx(j, i) - dataUy(j, i);
            dataUMAG(j, i) = sqrt(dataV(j, i)*dataV(j, i) + dataU(j, i)*dataU(j, i));
        end
    end

    % Data reduction by factors of sx (or sx1) and sy. This is necessary
    % if the quiver plot otherwise would produce more arrows than can
    % be properly visualized.
    X = dataX(1:sy:end,1:sx:end);
    Y = dataY(1:sy:end,1:sx:end);
    X1 = dataX(1:sy:end,1:sx1:end);
    Y1 = dataY(1:sy:end,1:sx1:end);
    U = dataU(1:sy:end,1:sx1:end);
    V = dataV(1:sy:end,1:sx1:end);
    Nx = dataNx(1:sy:end,1:sx:end);
    Ny = dataNy(1:sy:end,1:sx:end);
    %---------------------------------------------------------------------%

    % The director has a head-tail symmetry, i.e. sign ambiguity.
    % Therefore, to plot the director field, I construct two vector fields:
    % one pointing in the +(nx, ny) direction and the other in the (-nx,
    % ny) direction. Plotting both fields without arrow heads will then
    % generate a field of rods representing the director.
    dimYr = ceil(dimY/sy); dimXr = ceil(dimX/sx);
    Xr = zeros(2*dimYr, 2*dimXr);
    Yr = zeros(2*dimYr, 2*dimXr);
    N_xr = zeros(2*dimYr, 2*dimXr);
    N_yr = zeros(2*dimYr, 2*dimXr);

    Xr(1:dimYr,1:dimXr) = X; Xr(dimYr+1:2*dimYr,dimXr+1:2*dimXr) = X;
    Yr(1:dimYr,1:dimXr) = Y; Yr(dimYr+1:2*dimYr,dimXr+1:2*dimXr) = Y;
    N_xr(1:dimYr,1:dimXr) = Nx(1:dimYr,1:dimXr); N_xr(dimYr+1:2*dimYr,dimXr+1:2*dimXr) = -Nx(1:dimYr,1:dimXr);
    N_yr(1:dimYr,1:dimXr) = Ny(1:dimYr,1:dimXr); N_yr(dimYr+1:2*dimYr,dimXr+1:2*dimXr) = -Ny(1:dimYr,1:dimXr);
    %---------------------------------------------------------------------%

    set(gcf, 'Color','black');
    t = tiledlayout(2,1);

    nexttile
    contourf(dataX,dataY,dataS,100,'LineColor','none')
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    xlim([-0.5*width 0.5*width-offset]) 
    ylim([0 scale*height])
    daspect([ar 1 1])
    cb = colorbar;
    cb.Color = [0.9,0.9,0.9];
    %caxis([-0.35 0.35])
    %ax = gca
    %ax.FontSize = 1;
    %set(gca,'xtick',[],'ytick',[])
    set(gca,'FontSize',18)
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
    s_temp = "{\color{white}Nematic director field}";
    title(s_temp, 'Color','white','FontSize',18)
    %title('Defect tracking', 'Color','white','FontSize',18)
    hold on
    quiver(Xr,Yr,N_xr,N_yr,0.55,'ShowArrowHead',0,'LineWidth',0.9,'Color',[0.1,0.1,0.1])
    hold off
    
    
    nexttile
    contourf(dataX,dataY,dataUMAG, 100,'LineColor','none')
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    xlim([-0.5*width 0.5*width-offset]) 
    ylim([0 scale*height])
    daspect([1 1 1])
    set(gca,'FontSize',18)
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
    %cb3 = colorbar('south')
    %caxis([-0.046 0.046])
    set(gca,'FontSize',18)
    title('Flow velocity magnitude', 'Color','white')
    hold on
    quiver(X1,Y1,U,V,'LineWidth',0.75,'Color','black')
    hold off

    set(gca,'TickLength',[0.0025, 0.0025])
    cb = colorbar;
    cb.Color = [0.9,0.9,0.9];
    %cb3 = colorbar('south')
    %caxis([-0.2 0.2])
    set(gca,'FontSize',18)
    title('Vorticity', 'Color','white')
    hold on
    quiver(X1,Y1,U,V,'LineWidth',0.75,'Color','black')
    hold off

    %set(gca, 'color','black')
    export_fig("./snapshot.png")
    %saveas(gcf,'snapshot__S0-1_h-7.png');