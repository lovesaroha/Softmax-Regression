"use-strict";
/*  Love Saroha
    lovesaroha1994@gmail.com (email address)
    https://www.lovesaroha.com (website)
    https://github.com/lovesaroha  (github)
*/

// Themes.
const themes = [
    {
        normal: "#5468e7",
        dark: "#4353b9",
        light: "#98a4f1",
        veryLight: "#eef0fd"
    }, {
        normal: "#e94c2b",
        dark: "#ba3d22",
        veryLight: "#fdedea",
        light: "#f29480"
    }
];
// Choose random color theme.
let colorTheme = themes[Math.floor(Math.random() * themes.length)];

// This function set random color theme.
function setTheme() {
    // Change css values.
    document.documentElement.style.setProperty("--primary", colorTheme.normal);
    document.documentElement.style.setProperty("--primary-light", colorTheme.light);
    document.documentElement.style.setProperty("--primary-dark", colorTheme.dark);
}

// Set random theme.
setTheme();

// Get canvas info from DOM
var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');

// Default values.
var epochs = 200;

// Inputs matrix.
let inputs = tf.tensor([[10 / 400, 30 / 400], [30 / 400, 40 / 400], [50 / 400, 60 / 400], [90 / 400, 80 / 400], [210 / 400, 200 / 400] , [110 / 400, 100 / 400], [120 / 400, 110 / 400], [160 / 400, 120 / 400], [140 / 400, 130 / 400], [240 / 400, 220 / 400], [280 / 400, 320 / 400], [300 / 400, 300 / 400], [300 / 400, 380 / 400]]);
let outputs = tf.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0] , [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]);

// Show result.
function showResult() {
    let ip = [];
    // Results.
    for (let i = 20; i < 380; i = i + 20) {
        for (let j = 20; j < 380; j = j + 20) {
            ip.push([i / 400, j / 400]);
        }
    }
    ip = tf.tensor(ip);
    let p = model.predict(ip).dataSync();
    let m = 0;
    for (let i = 20; i < 380; i = i + 20) {
        for (let j = 20; j < 380; j = j + 20) {
            if (p[m] > p[m + 1] && p[m] > p[m + 2]) {
                ctx.fillStyle = colorTheme.light;
                ctx.fillRect(i + 2, j + 2, 18, 18);
            } else if (p[m + 1] > p[m] && p[m + 1] > p[m + 2]) {
                ctx.fillStyle = colorTheme.normal;
                ctx.fillRect(i + 2, j + 2, 18, 18);
            } else {
                ctx.fillStyle = colorTheme.dark;
                ctx.fillRect(i + 2, j + 2, 18, 18);
            }
            m += 3;
        }
    }
}

// Model.
const model = tf.sequential({
    layers: [
        tf.layers.dense({ inputShape: [2], units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 3, activation: 'softmax' }),
    ]
});

// Compile model.
model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

// On batch end show loss.
function onBatchEnd(batch, logs) {
    document.getElementById("loss_id").innerHTML = logs.loss.toFixed(2);
}


// Draw function.
let count = 0;
function draw() {
    model.fit(inputs, outputs, {
        epochs: 1,
        batchSize: 1,
        callbacks: { onBatchEnd }
    }).then(info => {
        count++;
        showResult();
        document.getElementById("epochs_id").innerHTML = count;
        if (count < epochs) {
            window.requestAnimationFrame(draw);
        }
    });
}
draw();



