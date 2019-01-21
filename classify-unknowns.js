const bci = require('bcijs');

(async () => {
    // Load training data
    let feetTraining = await bci.loadCSV('data/feet-training.csv');
    let rightTraining = await bci.loadCSV('data/righthand-training.csv');

    // Project it with CSP
    let cspParams = bci.cspLearn(feetTraining, rightTraining);

    // Compute training data features
    let featuresFeetTraining = computeFeatures(cspParams, feetTraining);
    let featuresRightTraining = computeFeatures(cspParams, rightTraining);

    // Learn an LDA classifier
    let ldaParams = bci.ldaLearn(featuresFeetTraining, featuresRightTraining);

    // Load testing data
    let feetTesting = await bci.loadCSV('data/feet-testing.csv');
    let rightTesting = await bci.loadCSV('data/righthand-testing.csv');

    // Compute testing data features
    let featuresFeetTesting = computeFeatures(cspParams, feetTesting);
    let featuresRightTesting = computeFeatures(cspParams, rightTesting);

    // Classify testing data
    let classify = (feature) => {
        let projection = bci.ldaProject(ldaParams, feature);
        // Filter out values between -0.5 and 0.5 as unknown classes
        if(projection < -0.5) return 0;
        if(projection > 0.5) return 1;
        return -1;
    }
    let feetPredictions = featuresFeetTesting.map(classify).filter(value => value != -1);
    let rightPredictions = featuresRightTesting.map(classify).filter(value => value != -1);

    // Evaluate the classifer
    let feetActual = new Array(feetPredictions.length).fill(0);
    let rightActual = new Array(rightPredictions.length).fill(1);

    let predictions = feetPredictions.concat(rightPredictions);
    let actual = feetActual.concat(rightActual);

    let confusionMatrix = bci.confusionMatrix(predictions, actual);

    let bac = bci.balancedAccuracy(confusionMatrix);

    let featureCount = featuresFeetTesting.length + featuresRightTesting.length;
    let percentUnknowns = (featureCount - predictions.length) / featureCount;

    console.log('confusion matrix');
    console.log(bci.toTable(confusionMatrix));
    console.log('balanced accuracy');
    console.log(bac);
    console.log('percent unknown');
    console.log(percentUnknowns);
})();

function computeFeatures(cspParams, eeg){
    let epochSize = 64; // About a fourth of a second per feature
    let trialLength = 750; // Each set of 750 samples is from a different trial

    let features = bci.windowApply(eeg, trial => {  
        // Apply CSP over each 64 sample window with a 50% overlap between windows
        return bci.windowApply(trial, epoch => {
            // Project the data with CSP and select the 16 most relevant signals
            let cspSignals = bci.cspProject(cspParams, epoch, 16);
            // Use the log of the variance of each signal as a feature vector
            return bci.features.logvar(cspSignals, 'columns');
        }, epochSize, epochSize / 2);
    }, trialLength, trialLength);

    // Concat the features from each trial
    return [].concat(...features);
}
