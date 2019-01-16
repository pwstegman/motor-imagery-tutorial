% Mat files from http://bnci-horizon-2020.eu/database/data-sets "Four class motor imagery (001-2014)"
% File description: https://lampx.tugraz.at/~bci/database/001-2014/description.pdf
% Tested on GNU Octave 4.2.2

%{
Uses files 'A01T' (training data) and 'A01E' (evaluation data) from http://bnci-horizon-2020.eu/database/data-sets.
These are .mat files which contain data from multiple sessions and trials. I've written this script in GNU Octave
(compatible with MATLAB as well) to extract the EEG data for each of the four classes to CSV files.
%}

training = load('data/A01T.mat');
evaluation = load('data/A01E.mat');

for i = 1:2
  if i == 1
    data = training.data;
  else
    data = evaluation.data;
  end
  
  % Store EEG for each of the 4 motor imagery classes
  classData = cell(1, 4);

  numSessions = size(data, 2);

  % First 3 sessions are EOG sessions, so starting at 4
  for session = 4:numSessions
    numSignals = size(data{session}.X, 1);
    numTrials = size(data{session}.y, 1);
    signalsPerTrial = numSignals / numTrials;

    for trial = 1:numTrials
      artifact = data{session}.artifacts(trial);
      if artifact
        continue;
      end
      % First 3 seconds at 250 hz are prompts, so adding 750 to skip
      start = round((trial - 1) * signalsPerTrial) + 750;
      % Extract the 3 seconds of motor imagery from the 22 EEG channels
      imagery = data{session}.X(start:start+749, 1:22);
      
      classification = data{session}.y(trial);
      classData{classification} = [classData{classification}; imagery];
    end
  end
  
  suffixes = {'training' 'testing'};
    
  csvwrite(['data/lefthand-' suffixes{i} '.csv'], classData{1});
  csvwrite(['data/righthand-' suffixes{i} '.csv'], classData{2});
  csvwrite(['data/feet-' suffixes{i} '.csv'], classData{3});
  csvwrite(['data/tongue-' suffixes{i} '.csv'], classData{4});
end
