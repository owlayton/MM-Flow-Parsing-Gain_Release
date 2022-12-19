function FlowParsingGain(nameValueArgs)
  %FlowParsingGain depth+self-motion gain mathematical model
  arguments
    nameValueArgs.path2Data char = 'exp2SubData.mat'
    nameValueArgs.gain_samps = 0.01:0.01:2
    nameValueArgs.objRelHeight = -1.2; % m
    nameValueArgs.relZ = 7.4 % Object-Observer relative depth (m)
    nameValueArgs.observerSpd = 0.8; % m/sec
    nameValueArgs.objectSpd = 2.5; % m/s
    nameValueArgs.f = 0.01; % focal length cm
    nameValueArgs.export logical = false
  end
  close all hidden;

  % Extract all subdirectories in the data directory
  load(nameValueArgs.path2Data, "subData");

  % Define helper vars for experiment conditions
  conds = fieldnames(subData);
  numConds = length(conds);
  numSubs = size(subData.moving_empty.judgedAngleSign, 2);

  % Save the histogram for each subject in each condition
  subModelHists = cell(numConds, numSubs);

  % Iterate through conditions
  for cond = 1:numConds
    fprintf('The condition is %s\n', conds{cond});

    condData = subData.(conds{cond});

    % Iterate through subjects
    for subject = 1:numSubs
      fprintf('    The subject is %d\n', subject);

      % Approach/retreat estimate (dependent variable)
      %  Approaching object coded as -1, retreating object coded as +1.
      subResponse = condData.judgedAngleSign(:, subject);

      % X/Z-component of object translation vector (independent variable)
      % Approaching object coded as -1, retreating object coded as +1.
      actTrajAngleSigned = condData.trueAngle(:, subject);
      txObj = nameValueArgs.objectSpd*cosd(actTrajAngleSigned);
      tzObj = nameValueArgs.objectSpd*sind(actTrajAngleSigned);

      % Remove NaNs
      nanI = isnan(subResponse);
      subResponse = subResponse(~nanI);
      tzObj = tzObj(~nanI);

      % Compute model match to human binary staircase judgments
      subModelHists{cond, subject} = sampleGsGzSurface(txObj, tzObj, subResponse, ...
        tzs=nameValueArgs.observerSpd, y=nameValueArgs.objRelHeight, relZ=nameValueArgs.relZ, f=nameValueArgs.f, ...
        GzInterval=nameValueArgs.gain_samps, GsInterval=nameValueArgs.gain_samps);
    end
    disp('done');
  end

  % Combine counts across subjects into a cross-subject histogram of number of times model matches human object sign
  posteriorSurf = computeCrossSubjectHistogram(subModelHists);
  % Plot cross-subject posterior/histogram of counts
  posteriorPlots(posteriorSurf, nameValueArgs.gain_samps);

  % Save off PDF of figure
  if nameValueArgs.export
    set(gcf,'PaperType', 'tabloid');
    saveas(gcf, 'posterior_plots.pdf')
  end
end

function normSurf = sampleGsGzSurface(txt, tzt, subResp, nameValueArgs)
  arguments
    txt (:, 1) double % object x translational component
    tzt (:, 1) double % object z translational component
    subResp (:, 1) double % Subject -1 / +1 approach retreat judgments for 40 trials in block
    nameValueArgs.tzs = 0.8 % Observer self-motion speed in z
    nameValueArgs.y = -1.2 % Observer-relative height of object
    nameValueArgs.f = 0.01 % Observer focal length
    nameValueArgs.relZ = 7.4 % Object-Observer relative depth (m)
    nameValueArgs.GzInterval = 0.01:0.01:2 % Model gz gain factor samples
    nameValueArgs.GsInterval = 0.01:0.01:2 % Model gs gain factor samples
  end

  GzInterval = nameValueArgs.GzInterval;
  GsInterval = nameValueArgs.GsInterval;
  tzs = nameValueArgs.tzs;
  y = nameValueArgs.y / nameValueArgs.relZ;

  % Net relative z translation between observer and object
  tzr = tzs - tzt;

  % Grid of matches between subject and model recovered object signs
  % columns (x): gs (observer self-motion gain)
  % rows (y): gz (depth gain)
  normSurf = zeros([length(GzInterval), length(GsInterval)]);

  % Generate histogram: For each (Gz, Go) sample pair, plug in model to get recovered object angle in retinal
  % coordinates then compare its sign with the one judged by the subject on a given trial.
  % At entry (Gz, Go) of surface, we tally number of trials where model produces consistent sign with human subjects
  for j = 1:length(GsInterval)
    for i = 1:length(GzInterval)
      numerator = y*(tzr - GsInterval(j)/GzInterval(i)*tzs);
      denominator = nameValueArgs.f*txt;
      funPredVals = atan(numerator./denominator);
      normSurf(i,j) = sum(sign(funPredVals) == sign(subResp));
    end
  end
end

function posteriorSurf = computeCrossSubjectHistogram(subModelHists)
  [numConds, numSubs] = size(subModelHists);

  % Shape: n_gz x n_gs x nConds
  posteriorSurf = zeros([size(subModelHists{1,1}), numConds]);
  % Combine counts across subjects into a cross-subject histogram of number of times model matches human object sign
  for c = 1:numConds
    for s = 1:numSubs
      posteriorSurf(:,:,c) = posteriorSurf(:,:,c) + subModelHists{c, s};
    end
  end
end

function posteriorPlots(posteriorSurf, gain_samps)
  f = figure(1);
  f.Position = [f.Position(1:2), 1.5*f.Position(3:4)];
  f.Position(4) = f.Position(3);
  % 2x2 tiled layout
  t = tiledlayout(2, 2, TileSpacing="compact");
  txt = title(t, 'Model posteriors fit to Exp 2 data');
  txt.FontSize = 20;


  % A) Real Untex raw posterior surface across subjects
  nexttile(1);
  [X, Y] = meshgrid(gain_samps, gain_samps);
  contourf(X, Y, posteriorSurf(:,:,1), LineStyle="none");
  ticks = [min(posteriorSurf(:,:,1), [], "all"), max(posteriorSurf(:,:,1), [], "all")-7];
  colorbar(Ticks=ticks, TickLabels=[0, 1]);

  xline(1);
  yline(1);
  ylabel('Depth gain (Gz)')
  title('Priors: uniform', 'Real Untextured')
  xticks(0:0.5:2)
  yticks(0:0.5:2)

  ax = gca;
  ax.FontSize = 16;
  ax.TickDir = 'both';
  ax.TickLength = [0.02 0.035];


  % B) Real Untex tight prior on Gz = 1
  nexttile(2);
  gz_1_prior = gaussianPrior(gain_samps, 1);
  % Weight y (rows) values of surface by prior
  posterior_gz_1_prior = posteriorSurf(:,:,1) .* reshape(gz_1_prior, [numel(gz_1_prior), 1]);

  contourf(X, Y, posterior_gz_1_prior, LineStyle="none")

  xline(1);
  title('Prior: Gauss prior at Gz=1', 'Real Untextured')
  xticks(0:0.25:2)
  yticks(0.6:0.1:1.1)
  ylim([0.65, 1.05])

  ax = gca;
  ax.FontSize = 16;
  ax.TickDir = 'both';
  ax.TickLength = [0.02 0.035];
  ax.XTickLabelRotation = 0;


  % C) Real Tex tight prior on Gz = 1
  nexttile(3);
  gz_1_prior = gaussianPrior(gain_samps, 1);
  % Weight y (rows) values of surface by prior
  posterior_gz_1_prior = posteriorSurf(:,:,2) .* reshape(gz_1_prior, [numel(gz_1_prior), 1]);

  contourf(X, Y, posterior_gz_1_prior, LineStyle="none")

  xline(1);
  ylabel('Depth gain (Gz)')
  xlabel('Self-motion gain (Gs)')
  title('Prior: Gauss prior at Gz=1', 'Real Textured')
  xticks(0:0.25:2)
  yticks(0.6:0.1:1.1)
  ylim([0.65, 1.05])

  ax = gca;
  ax.FontSize = 16;
  ax.TickDir = 'both';
  ax.TickLength = [0.02 0.035];
  ax.XTickLabelRotation = 0;


  % D) Prior on Gs: uniform < 1, 0 > 1. Also, prior at Gz=0.7, not including 1
  % Prior: Gs uniform <= 1, 0 if > 1
  % Find index of Gs = 1.0
  nexttile(4);
  gs_thres_prior = stepPrior(gain_samps, 1);

  % Prior: Gz=0.7, not including 1
  gz_7_prior = gaussianPrior(gain_samps, 0.7);

  % Weight x (cols) values of surface by priors
  posterior_joint_prior = posteriorSurf(:,:,1) .* reshape(gz_7_prior, [numel(gz_7_prior), 1]) ...
    .* reshape(gs_thres_prior, [1, numel(gs_thres_prior)]);

  contourf(X, Y, posterior_joint_prior, LineStyle="none")

  xline(1);
  yline(1);
  xlabel('Self-motion gain (Gs)')
  title('Priors: Gauss Gz=0.7, step Gs', 'Real Untextured')
  xticks(0:0.25:2)
  yticks(0.6:0.1:1.1)
  ylim([0.65, 1.05])

  ax = gca;
  ax.FontSize = 16;
  ax.TickDir = 'both';
  ax.TickLength = [0.02 0.035];
  ax.XTickLabelRotation = 0;
end

function prior = gaussianPrior(gain_samps, mu, sigma, nameValueArgs)
  arguments
    gain_samps (:, 1) double
    mu double
    sigma double = 0.025
    nameValueArgs.normalize logical = true
  end

  % Find nearest index of Gz = mu
  [~, gz_1_ind] = min(abs(gain_samps - mu));
  prior = normpdf(gain_samps, gain_samps(gz_1_ind), sigma);

  if nameValueArgs.normalize
    % Normalize prior to sum to 1
    prior = prior ./ sum(prior);
  end
end

function prior = stepPrior(gain_samps, stepValue, nameValueArgs)
  arguments
    gain_samps (:, 1) double
    stepValue double
    nameValueArgs.normalize logical = true
  end

  [~, crit_ind] = min(abs(gain_samps - stepValue));
  prior = zeros(numel(gain_samps), 1);
  prior(1:crit_ind) = 1;

  if nameValueArgs.normalize
    % Normalize prior to sum to 1
    prior = prior ./ sum(prior);
  end
end