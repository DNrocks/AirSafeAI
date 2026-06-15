let session;

let map;
let marker;
let circleLayer;

// API KEY

const API_KEY =
    "8385aa7e3ee75e99c7c458ce685c017f";

// STANDARD SCALER VALUES

const MEAN = [
    248.43847634,
    148.65499661,
    100.22371356,
    102.29344544,
    49.45683794,
    149.31243055,
    14.97549957,
    54.77685271,
    9.98917734
];

const SCALE = [
    144.76517985,
    85.69112612,
    58.0916127,
    57.7082093,
    28.52787389,
    86.52679429,
    14.4818212,
    26.01854661,
    5.77645285
];

// LOAD MODEL

async function loadModel() {

    try {

        session = await ort.InferenceSession.create(
            "model/airsafe.onnx"
        );

        console.log("ONNX model loaded");

    } catch (error) {

        console.error(error);
    }
}

loadModel();

// MAIN FUNCTION

async function analyzeAir() {

    document.getElementById("loading").innerHTML =
        "Fetching live environmental data...";

    try {

        navigator.geolocation.getCurrentPosition(

            async (position) => {

                const lat =
                    position.coords.latitude;

                const lon =
                    position.coords.longitude;

                // FETCH AIR DATA

                const airResponse =
                    await fetch(

                        `https://api.openweathermap.org/data/2.5/air_pollution?lat=${lat}&lon=${lon}&appid=${API_KEY}`

                    );

                const airData =
                    await airResponse.json();

                // FETCH WEATHER DATA

                const weatherResponse =
                    await fetch(

                        `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${API_KEY}&units=metric`

                    );

                const weatherData =
                    await weatherResponse.json();

                // FEATURES

                const AQI =
                    airData.list[0].main.aqi * 50;

                const PM10 =
                    airData.list[0].components.pm10;

                const PM25 =
                    airData.list[0].components.pm2_5;

                const NO2 =
                    airData.list[0].components.no2;

                const SO2 =
                    airData.list[0].components.so2;

                const O3 =
                    airData.list[0].components.o3;

                const Temperature =
                    weatherData.main.temp;

                const Humidity =
                    weatherData.main.humidity;

                const WindSpeed =
                    weatherData.wind.speed;

                // RAW INPUTS

                const rawData = [
                    AQI,
                    PM10,
                    PM25,
                    NO2,
                    SO2,
                    O3,
                    Temperature,
                    Humidity,
                    WindSpeed
                ];

                // SCALE INPUTS

                const scaledData =
                    rawData.map((value, index) => {

                        return (
                            (value - MEAN[index]) /
                            SCALE[index]
                        );

                    });

                // CREATE INPUT

                const inputData =
                    new Float32Array(scaledData);

                // CREATE TENSOR

                const tensor =
                    new ort.Tensor(
                        "float32",
                        inputData,
                        [1, 9]
                    );

                // RUN MODEL

                const feeds = {
                    X: tensor
                };

                const results =
                    await session.run(feeds);

                // LABEL

                const labelTensor =
                    results.label;

                const predictedClass =
                    Number(labelTensor.cpuData[0]);

                // LABELS

                const labels = [
                    "Very Low",
                    "Low",
                    "Moderate",
                    "High",
                    "Severe"
                ];

                const risk =
                    labels[predictedClass];

                // DETAILED ANALYSIS

                let advice = "";

                let pollutionSummary = "";

                if (PM25 > 120) {

                    pollutionSummary += `
                    PM2.5 levels are critically high and may
                    penetrate deep into lung tissue.
                    `;
                }
                else if (PM25 > 60) {

                    pollutionSummary += `
                    PM2.5 concentration is elevated and may
                    affect sensitive individuals.
                    `;
                }

                if (NO2 > 100) {

                    pollutionSummary += `
                    Nitrogen dioxide levels suggest strong
                    traffic or combustion-related pollution.
                    `;
                }

                if (O3 > 180) {

                    pollutionSummary += `
                    Ozone concentration is unusually high,
                    indicating strong photochemical smog activity.
                    `;
                }

                if (Humidity > 75) {

                    pollutionSummary += `
                    High humidity may intensify discomfort
                    and worsen pollutant retention near ground level.
                    `;
                }

                if (WindSpeed < 2) {

                    pollutionSummary += `
                    Low wind speed reduces pollutant dispersion,
                    allowing contaminants to accumulate.
                    `;
                }

                if (Temperature > 35) {

                    pollutionSummary += `
                    High temperature may accelerate ozone formation
                    and heat-related respiratory stress.
                    `;
                }

                // RISK-BASED ANALYSIS

                if (risk === "Severe") {

                    advice = `
                    ⚠️ Air quality conditions are hazardous.

                    Outdoor exposure should be minimized,
                    especially for children, elderly individuals,
                    and people with respiratory conditions.

                    Extended outdoor exercise is strongly discouraged.

                    ${pollutionSummary}
                    `;

                }
                else if (risk === "High") {

                    advice = `
                    ⚠️ Air quality is unhealthy.

                    Sensitive individuals should reduce
                    prolonged outdoor exposure.

                    Wearing a filtration mask outdoors
                    is recommended in congested areas.

                    ${pollutionSummary}
                    `;

                }
                else if (risk === "Moderate") {

                    advice = `
                    Air quality is acceptable for most people,
                    though sensitive groups should remain cautious.

                    ${pollutionSummary}
                    `;

                }
                else {

                    advice = `
                    ✅ Air quality conditions are currently stable.

                    Pollutant concentrations remain within
                    relatively safer operational thresholds.
                    `;
                }

                // MAP

                if (!map) {

                    map = L.map('map').setView(
                        [lat, lon],
                        11
                    );

                    L.tileLayer(
                        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                        {
                            attribution:
                                '&copy; OpenStreetMap contributors'
                        }
                    ).addTo(map);

                } else {

                    map.setView([lat, lon], 11);
                }

                // REMOVE OLD MARKER

                if (marker) {
                    marker.remove();
                }

                // REMOVE OLD CIRCLE

                if (circleLayer) {
                    circleLayer.remove();
                }

                // RISK COLOR

                let riskColor = "green";

                if (risk === "Severe") {

                    riskColor = "red";

                } else if (risk === "High") {

                    riskColor = "orange";

                } else if (risk === "Moderate") {

                    riskColor = "yellow";
                }

                // MARKER

                marker = L.marker([lat, lon])
                    .addTo(map)
                    .bindPopup(`

                        <b>${weatherData.name}</b>
                        <br>
                        AQI: ${AQI.toFixed(1)}
                        <br>
                        Risk: ${risk}
                        <br>
                        PM2.5: ${PM25.toFixed(1)}

                    `)
                    .openPopup();

                // DANGER CIRCLE

                circleLayer = L.circle([lat, lon], {

                    color: riskColor,

                    fillColor: riskColor,

                    fillOpacity: 0.25,

                    radius: AQI * 12

                }).addTo(map);

                setTimeout(() => {
                    map.invalidateSize();
                }, 100);

                // DISPLAY

                document.getElementById("loading").innerHTML =
                    "";

                document.getElementById("result").innerHTML = `

                    <div class="result-card">

                        <h2 class="${risk.toLowerCase().replace(' ', '-')}">
                            ${risk}
                        </h2>

                        <p>
                            📍 <b>${weatherData.name}</b>
                        </p>

                        <div class="metrics">

                            <div class="metric">
                                <div class="metric-title">
                                    AQI
                                </div>
                                <div class="metric-value">
                                    ${AQI.toFixed(1)}
                                </div>
                            </div>

                            <div class="metric">
                                <div class="metric-title">
                                    PM2.5
                                </div>
                                <div class="metric-value">
                                    ${PM25.toFixed(1)}
                                </div>
                            </div>

                            <div class="metric">
                                <div class="metric-title">
                                    Temperature
                                </div>
                                <div class="metric-value">
                                    ${Temperature}°C
                                </div>
                            </div>

                            <div class="metric">
                                <div class="metric-title">
                                    Humidity
                                </div>
                                <div class="metric-value">
                                    ${Humidity}%
                                </div>
                            </div>

                        </div>

                        <hr>

                        <p>
                            <b>Environmental Analysis</b>
                        </p>

                        <p>
                            ${advice}
                        </p>

                    </div>

                `;

            },

            () => {

                document.getElementById("loading").innerHTML =
                    "Location access denied.";

            }

        );

    } catch (error) {

        console.error(error);

        document.getElementById("loading").innerHTML =
            "Prediction failed.";

    }
}