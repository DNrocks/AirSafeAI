/* ============================================================
   AIRSAFE AI
   Main application
   ============================================================ */

"use strict";


/* ============================================================
   CONFIGURATION
   ============================================================ */

const CONFIG = {
    
    MAP: {
    CARTO_API_KEY: "cb1_322k_1_4d6129e6d225172ebc4861c0",

    TILE_URL:
        "https://{s}.basemaps.cartocdn.com/"
        + "rastertiles/voyager/{z}/{x}/{y}.png"
    },

    /*
     * Open-Meteo
     *
     * No API key required for normal student/non-commercial use.
     */

    GEOCODING_URL:
        "https://geocoding-api.open-meteo.com/v1/search",

    AIR_QUALITY_URL:
        "https://air-quality-api.open-meteo.com/v1/air-quality",

    WEATHER_URL:
        "https://api.open-meteo.com/v1/forecast",


    /*
     * ONNX models
     *
     * Keep the model files in the same directory as index.html.
     */

    CLASSIFIER_MODEL:
        "airsafe_classifier.onnx",

    REGRESSOR_MODEL:
        "airsafe_regressor.onnx",

    ANOMALY_MODEL:
        "anomaly_detector.onnx",


    /*
     * These MUST remain in exactly this order.
     *
     * This is the order used when the models were trained.
     */

    FEATURES: [
        "AQI",
        "PM10",
        "PM2_5",
        "NO2",
        "SO2",
        "O3",
        "Temperature",
        "Humidity",
        "WindSpeed"
    ],


    /*
     * Classification labels.
     */

    CLASS_LABELS: {
        0: "Very Low",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "Very High"
    }
};

/* ============================================================
   GEMINI AI
   ============================================================ */

const GEMINI_CONFIG = {

    API_KEY:
        "AQ.Ab8RN6JAAzjsHoZzOLMd7wo0v7VYiuqqZUAzmRNtsAUdrSZAsg",

    MODEL:
        "gemini-3.5-flash-lite",

    API_URL:
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + "gemini-3.5-flash-lite:generateContent"
};

/* ============================================================
   APPLICATION STATE
   ============================================================ */

const state = {

    location: {
        name: null,
        latitude: null,
        longitude: null,
        country: null
    },

    environment: {
        AQI: null,
        PM10: null,
        PM2_5: null,
        NO2: null,
        SO2: null,
        O3: null,
        Temperature: null,
        Humidity: null,
        WindSpeed: null
    },

    models: {
        classifier: null,
        regressor: null,
        anomaly: null
    },

    predictions: {
        classification: null,
        probabilities: {},
        confidence: null,
        regression: null,
        anomaly: null,
        anomalyScore: null
    },

    initialized: false,
    dataLoaded: false
};


/* ============================================================
   DOM HELPERS
   ============================================================ */

function $(id) {
    return document.getElementById(id);
}


function setText(id, value) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent = value;
}


function showElement(id, show = true) {

    const element = $(id);

    if (!element) {
        return;
    }

    element.style.display = show ? "" : "none";
}


/* ============================================================
   STATUS / ERROR HANDLING
   ============================================================ */

function setSystemStatus(main, detail, isError = false) {

    const mainElement = $("systemStatusMain");
    const detailElement = $("systemStatusDetail");
    const dot = $("systemStatusDot");

    if (mainElement) {
        mainElement.textContent = main;
    }

    if (detailElement) {
        detailElement.textContent = detail;
    }

    if (dot) {

        dot.style.background =
            isError ? "var(--red)" : "var(--green)";

        dot.style.boxShadow =
            isError
                ? "0 0 12px rgba(248,113,113,0.8)"
                : "0 0 12px rgba(52,211,153,0.8)";
    }
}


function showError(message) {

    console.error("AirSafe:", message);

    setSystemStatus(
        "SYSTEM ERROR",
        message,
        true
    );
}

/* ============================================================
   GEMINI API
   ============================================================ */

async function callGemini(prompt) {

    if (
        !GEMINI_CONFIG.API_KEY ||
        GEMINI_CONFIG.API_KEY.includes(
            "PASTE_YOUR_GEMINI_API_KEY_HERE"
        )
    ) {
        throw new Error(
            "Gemini API key has not been configured."
        );
    }

    const response = await fetch(
        GEMINI_CONFIG.API_URL,
        {
            method: "POST",

            headers: {
                "Content-Type":
                    "application/json",

                "x-goog-api-key":
                    GEMINI_CONFIG.API_KEY
            },

            body: JSON.stringify({
                contents: [
                    {
                        role: "user",

                        parts: [
                            {
                                text: prompt
                            }
                        ]
                    }
                ],

                generationConfig: {
                    temperature: 0.3,
                    maxOutputTokens: 500
                }
            })
        }
    );

    if (!response.ok) {

        let errorText = "";

        try {
            errorText =
                await response.text();
        } catch {
            errorText = "";
        }

        throw new Error(
            `Gemini request failed (${response.status})`
            + (errorText
                ? `: ${errorText}`
                : "")
        );
    }

    const data =
        await response.json();

    const text =
        data?.candidates?.[0]?.content?.parts
            ?.map(part => part.text || "")
            .join("")
            .trim();

    if (!text) {

        throw new Error(
            "Gemini returned an empty response."
        );
    }

    return text;
}
/* ============================================================
   GEMINI CONTEXT
   ============================================================ */

function buildAirSafeContext() {

    const environment =
        state.environment;

    const predictions =
        state.predictions;

    const probabilityText =
        Object.entries(
            predictions.probabilities || {}
        )
        .map(
            ([key, value]) =>
                `Class ${key}: ${(value * 100).toFixed(2)}%`
        )
        .join(", ");

    return `
AIRSAFE AI CURRENT ANALYSIS

Location:
${state.location.name || "Unknown"}
${state.location.admin1 || ""}
${state.location.country || ""}

Environmental conditions:
AQI: ${formatNumber(environment.AQI)}
PM10: ${formatNumber(environment.PM10)}
PM2.5: ${formatNumber(environment.PM2_5)}
NO2: ${formatNumber(environment.NO2)}
SO2: ${formatNumber(environment.SO2)}
O3: ${formatNumber(environment.O3)}
Temperature: ${formatNumber(environment.Temperature)} °C
Humidity: ${formatNumber(environment.Humidity)} %
Wind speed: ${formatNumber(environment.WindSpeed)} km/h

AIRSAFE ML OUTPUTS

Internal classifier class:
${predictions.classification}

Displayed safety classification:
${getSafetyClassification(
    predictions.classification
)}

Classifier probabilities:
${probabilityText}

Health Impact Score:
${predictions.regression !== null
    ? predictions.regression.toFixed(1)
    : "Unavailable"}

Anomaly status:
${predictions.anomaly || "Unavailable"}

Anomaly score:
${predictions.anomalyScore !== null
    ? predictions.anomalyScore.toFixed(4)
    : "Unavailable"}

IMPORTANT:
The ONNX models produced the numerical predictions above.
Do not change, reinterpret, or invent those predictions.
Gemini's role is to explain them in understandable language.
The Health Impact Score is a dataset-derived AI score,
not a medically validated probability or diagnosis.
`;
}
/* ============================================================
   SAFETY CLASSIFICATION
   ============================================================ */

function getSafetyClassification(classification) {

    const labels = {
        0: "Very Low",
        1: "Low",
        2: "Moderate"
    };

    if (
        classification === 0 ||
        classification === 1 ||
        classification === 2
    ) {
        return labels[classification];
    }

    return "3+ — Outside primary safety range";
}
/* ============================================================
   AI EXPLAINABILITY
   ============================================================ */

async function generateAIExplanation() {

    const reasoning =
        $("modelReasoning");

    if (!reasoning) {
        return;
    }

    if (!state.dataLoaded) {

        reasoning.textContent =
            "Analyse a location first to generate an AI explanation.";

        return;
    }

    reasoning.textContent =
        "Gemini is analysing the AirSafe model output...";

    try {

        const context =
            buildAirSafeContext();

        const prompt = `
You are the explainability layer for AirSafe AI.

Explain the following machine-learning result to a
school-project user.

${context}

Give a concise explanation in 2–4 short paragraphs.

Explain:
1. What the AirSafe classifier predicted.
2. Which environmental conditions are most relevant.
3. What the Health Impact Score means.
4. Whether the anomaly detector found an unusual pattern.

Do NOT claim that AirSafe provides a medical diagnosis.
Do NOT invent causal medical claims.
Do NOT change the model's numerical results.

Use plain but technically accurate language.
`;

        const explanation =
            await callGemini(prompt);

        reasoning.textContent =
            explanation;

    } catch (error) {

        console.error(
            "Gemini explainability failed:",
            error
        );

        reasoning.textContent =
            "Gemini explanation unavailable. "
            + error.message;
    }
}
/* ============================================================
   GENAI ASSISTANT
   ============================================================ */

async function handleGeminiAssistant() {

    const input =
        $("faqInput");

    const messages =
        $("faqMessages");

    const status =
        $("faqStatus");

    if (!input || !messages) {
        return;
    }

    const question =
        input.value.trim();

    if (!question) {
        return;
    }

    if (!state.dataLoaded) {

        if (status) {
            status.textContent =
                "Analyse a location before asking AirSafe AI.";
        }

        return;
    }

    addAssistantMessage(
        question,
        "user"
    );

    input.value = "";

    if (status) {
        status.textContent =
            "AirSafe AI is thinking...";
    }

    try {

        const context =
            buildAirSafeContext();

        const prompt = `
You are AirSafe AI Assistant.

You are assisting a user who is looking at
environmental data and machine-learning results.

${context}

USER QUESTION:
${question}

Answer the user's question directly.

Rules:
- Use the AirSafe data provided above.
- Never invent environmental measurements.
- Never change the ONNX prediction.
- Clearly distinguish model output from general explanation.
- Do not provide medical diagnosis.
- If the user asks whether conditions are dangerous,
  explain the environmental result cautiously.
- Keep the answer concise and understandable.
- You may explain AQI, pollutants, anomaly detection,
  model confidence, and Health Impact Score.
`;

        const answer =
            await callGemini(prompt);

        addAssistantMessage(
            answer,
            "bot"
        );

        if (status) {
            status.textContent =
                "Gemini connected";
        }

    } catch (error) {

        console.error(
            "Gemini assistant failed:",
            error
        );

        addAssistantMessage(
            "I couldn't reach Gemini right now. "
            + error.message,
            "bot"
        );

        if (status) {
            status.textContent =
                "Gemini request failed";
        }
    }
}
function addAssistantMessage(
    text,
    type
) {

    const container =
        $("faqMessages");

    if (!container) {
        return;
    }

    const message =
        document.createElement("div");

    message.className =
        `assistant-message ${type}`;

    const label =
        document.createElement("div");

    label.className =
        "message-label";

    label.textContent =
        type === "user"
            ? "YOU"
            : "AIRSAFE AI";

    const paragraph =
        document.createElement("p");

    paragraph.textContent =
        text;

    message.appendChild(label);
    message.appendChild(paragraph);

    container.appendChild(message);

    container.scrollTop =
        container.scrollHeight;
}
/* ============================================================
   INITIALIZATION
   ============================================================ */

document.addEventListener("DOMContentLoaded", async () => {

    console.log(
        "============================================================"
    );

    console.log("AIRSAFE AI INITIALIZING");

    console.log(
        "============================================================"
    );

    try {

        setSystemStatus(
            "INITIALIZING",
            "Loading AI models..."
        );

        await loadModels();
        await loadFeatureImportance();
        initializeMap();

        state.initialized = true;

        setSystemStatus(
            "AI READY",
            "Models loaded successfully"
        );

        console.log("AirSafe initialization complete.");

        attachEventListeners();
        attachPersonalisationListeners();

    } catch (error) {

        console.error(
            "Initialization failed:",
            error
        );

        showError(
            "AI models could not be loaded"
        );
    }
});


/* ============================================================
   EVENT LISTENERS
   ============================================================ */

function attachEventListeners() {

    const locationButton =
        $("locationSearchButton");

    const locationInput =
        $("locationInput");


    if (locationButton) {

        locationButton.addEventListener(
            "click",
            handleLocationSearch
        );
    }


    if (locationInput) {

        locationInput.addEventListener(
            "keydown",
            event => {

                if (event.key === "Enter") {

                    handleLocationSearch();
                }
            }
        );
    }
    const faqInput =
        $("faqInput");

    const faqSend =
        $("faqSend");

    if (faqSend) {

        faqSend.addEventListener(
            "click",
            handleGeminiAssistant
        );
    }

    if (faqInput) {

        faqInput.addEventListener(
            "keydown",
            event => {

                if (event.key === "Enter") {

                    event.preventDefault();

                    handleGeminiAssistant();
                }
            }
        );
    }
        /* --------------------------------------------------------
       SIMULATION LISTENERS
       -------------------------------------------------------- */

    const simulationButton =
        $("run-simulation-btn");

    const resetSimulationButton =
        $("reset-simulation-btn");


    if (simulationButton) {

        simulationButton.addEventListener(
            "click",
            runSimulation
        );
    }


    if (resetSimulationButton) {

        resetSimulationButton.addEventListener(
            "click",
            resetSimulation
        );
    }


    /*
     * Update the displayed value whenever
     * a slider moves.
     */

    SIMULATION_FEATURES.forEach(
        feature => {

            const input =
                $(feature.input);

            if (!input) {
                return;
            }

            input.addEventListener(
                "input",
                updateSimulationSliderDisplays
            );
        }
    );


    /*
     * Simulation starts from the real
     * environmental data whenever a
     * location has been analysed.
     */
}


/* ============================================================
   MODEL LOADING
   ============================================================ */

async function loadModels() {

    if (
        typeof ort === "undefined"
    ) {

        throw new Error(
            "ONNX Runtime Web is not loaded. " +
            "Check the ONNX Runtime script in index.html."
        );
    }


    console.log("Loading classifier...");

    state.models.classifier =
        await ort.InferenceSession.create(
            CONFIG.CLASSIFIER_MODEL
        );

    console.log(
        "Classifier loaded:",
        state.models.classifier.inputNames,
        state.models.classifier.outputNames
    );


    console.log("Loading regressor...");

    state.models.regressor =
        await ort.InferenceSession.create(
            CONFIG.REGRESSOR_MODEL
        );

    console.log(
        "Regressor loaded:",
        state.models.regressor.inputNames,
        state.models.regressor.outputNames
    );


    console.log("Loading anomaly detector...");

    state.models.anomaly =
        await ort.InferenceSession.create(
            CONFIG.ANOMALY_MODEL
        );

    console.log(
        "Anomaly detector loaded:",
        state.models.anomaly.inputNames,
        state.models.anomaly.outputNames
    );


    console.log(
        "All three ONNX models loaded successfully."
    );
}


/* ============================================================
   LOCATION SEARCH
   ============================================================ */

async function handleLocationSearch() {

    const input =
        $("locationInput");

    if (!input) {

        showError(
            "Location input was not found."
        );

        return;
    }


    const query =
        input.value.trim();


    if (!query) {

        showError(
            "Please enter a location."
        );

        return;
    }


    const button =
        $("locationSearchButton");


    if (button) {

        button.disabled = true;

        button.classList.add("loading");
    }


    setSystemStatus(
        "SEARCHING",
        `Finding ${query}...`
    );


    try {

        const location =
            await geocodeLocation(query);


        if (!location) {

            throw new Error(
                `Could not find "${query}".`
            );
        }


        state.location =
            location;


        updateLocationUI();

        updateMapLocation();

        setSystemStatus(
            "FETCHING DATA",
            "Retrieving live environmental conditions..."
        );


        await fetchEnvironmentalData();


        state.dataLoaded = true;


        updateEnvironmentUI();


        setSystemStatus(
            "RUNNING AI",
            "Running three ONNX models..."
        );


        await runAllModels();


        updatePredictionUI();

        generateAIExplanation();

        

        /*
        * Refresh simulation controls whenever
        * a new location/environment is loaded.
        */
        loadCurrentConditionsIntoSimulation();
        updateSimulationCurrentResults();

        setSystemStatus(
            "AI ACTIVE",
            "Live data analysed successfully"
        );


    } catch (error) {

        console.error(
            "Location/data pipeline failed:",
            error
        );

        showError(
            error.message ||
            "Unable to retrieve environmental data."
        );


    } finally {

        if (button) {

            button.disabled = false;

            button.classList.remove("loading");
        }
    }
}


/* ============================================================
   GEOCODING
   ============================================================ */

async function geocodeLocation(query) {

    const url =
        new URL(CONFIG.GEOCODING_URL);


    url.searchParams.set(
        "name",
        query
    );

    url.searchParams.set(
        "count",
        "1"
    );

    url.searchParams.set(
        "language",
        "en"
    );

    url.searchParams.set(
        "format",
        "json"
    );


    const response =
        await fetch(url);


    if (!response.ok) {

        throw new Error(
            `Geocoding request failed (${response.status}).`
        );
    }


    const data =
        await response.json();


    if (
        !data.results ||
        data.results.length === 0
    ) {

        return null;
    }


    const result =
        data.results[0];


    return {

        name:
            result.name,

        latitude:
            Number(result.latitude),

        longitude:
            Number(result.longitude),

        country:
            result.country || "",

        admin1:
            result.admin1 || "",

        timezone:
            result.timezone || ""
    };
}


/* ============================================================
   ENVIRONMENTAL DATA
   ============================================================ */

async function fetchEnvironmentalData() {

    const {
        latitude,
        longitude
    } = state.location;


    /*
     * We request all pollutants required by the model.
     */

    const airUrl =
        new URL(CONFIG.AIR_QUALITY_URL);


    airUrl.searchParams.set(
        "latitude",
        latitude
    );

    airUrl.searchParams.set(
        "longitude",
        longitude
    );


    airUrl.searchParams.set(
        "hourly",
        [
            "pm10",
            "pm2_5",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ].join(",")
    );


    airUrl.searchParams.set(
        "forecast_days",
        "1"
    );


    /*
     * Weather variables.
     */

    const weatherUrl =
        new URL(CONFIG.WEATHER_URL);


    weatherUrl.searchParams.set(
        "latitude",
        latitude
    );

    weatherUrl.searchParams.set(
        "longitude",
        longitude
    );


    weatherUrl.searchParams.set(
        "current",
        [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m"
        ].join(",")
    );


    const [
        airResponse,
        weatherResponse
    ] = await Promise.all([

        fetch(airUrl),

        fetch(weatherUrl)
    ]);


    if (!airResponse.ok) {

        throw new Error(
            `Air-quality request failed (${airResponse.status}).`
        );
    }


    if (!weatherResponse.ok) {

        throw new Error(
            `Weather request failed (${weatherResponse.status}).`
        );
    }


    const airData =
        await airResponse.json();


    const weatherData =
        await weatherResponse.json();


    /*
     * Open-Meteo returns hourly arrays.
     *
     * We use the first/current available value.
     */

    const air =
        airData.hourly;


    if (
        !air ||
        !air.time ||
        air.time.length === 0
    ) {

        throw new Error(
            "No air-quality data was returned."
        );
    }


    /*
     * Find the current hour rather than blindly
     * assuming a particular array index.
     */

    const currentIndex =
        findNearestCurrentIndex(
            air.time
        );


    const weather =
        weatherData.current;


    state.environment = {

        AQI:
            getArrayValue(
                air.us_aqi,
                currentIndex
            ),

        PM10:
            getArrayValue(
                air.pm10,
                currentIndex
            ),

        PM2_5:
            getArrayValue(
                air.pm2_5,
                currentIndex
            ),

        NO2:
            getArrayValue(
                air.nitrogen_dioxide,
                currentIndex
            ),

        SO2:
            getArrayValue(
                air.sulphur_dioxide,
                currentIndex
            ),

        O3:
            getArrayValue(
                air.ozone,
                currentIndex
            ),

        Temperature:
            Number(
                weather?.temperature_2m
            ),

        Humidity:
            Number(
                weather?.relative_humidity_2m
            ),

        WindSpeed:
            Number(
                weather?.wind_speed_10m
            )
    };


    console.log(
        "Environmental data:",
        state.environment
    );
}


/* ============================================================
   DATA HELPERS
   ============================================================ */

function getArrayValue(array, index) {

    if (
        !Array.isArray(array) ||
        index < 0 ||
        index >= array.length
    ) {

        return NaN;
    }


    const value =
        Number(array[index]);


    return Number.isFinite(value)
        ? value
        : NaN;
}


function findNearestCurrentIndex(times) {

    const now =
        Date.now();


    let bestIndex = 0;

    let smallestDifference =
        Infinity;


    times.forEach(
        (time, index) => {

            const timestamp =
                new Date(time).getTime();


            const difference =
                Math.abs(
                    timestamp - now
                );


            if (
                difference <
                smallestDifference
            ) {

                smallestDifference =
                    difference;

                bestIndex =
                    index;
            }
        }
    );


    return bestIndex;
}


/* ============================================================
   FEATURE VECTOR
   ============================================================ */

function buildFeatureVector() {

    const values =
        CONFIG.FEATURES.map(
            feature =>
                Number(
                    state.environment[feature]
                )
        );


    const invalidFeature =
        values.findIndex(
            value =>
                !Number.isFinite(value)
        );


    if (invalidFeature !== -1) {

        throw new Error(
            `Invalid model input: ${CONFIG.FEATURES[invalidFeature]}`
        );
    }


    return values;
}


/* ============================================================
   RUN ALL MODELS
   ============================================================ */

async function runAllModels() {

    const values =
        buildFeatureVector();


    console.log(
        "============================================================"
    );

    console.log(
        "MODEL INPUT VECTOR"
    );

    console.table(
        CONFIG.FEATURES.map(
            (feature, index) => ({

                Feature:
                    feature,

                Value:
                    values[index]
            })
        )
    );


    /*
     * IMPORTANT:
     *
     * Float32Array is required because the ONNX
     * models expect tensor(float).
     */

    const tensor =
        new ort.Tensor(
            "float32",
            Float32Array.from(values),
            [1, 9]
        );


    await runClassifier(tensor);

    await runRegressor(tensor);

    await runAnomalyDetector(tensor);


    console.log(
        "============================================================"
    );

    console.log(
        "MODEL RESULTS"
    );

    console.log(
        state.predictions
    );
}

/* ============================================================
   WHAT-IF ENVIRONMENTAL SIMULATION
   ============================================================ */

/*
 * Run all three ONNX models against a supplied feature vector.
 *
 * IMPORTANT:
 * This does NOT modify state.environment or state.predictions.
 * It is purely a hypothetical simulation.
 */
async function runSimulationModels(values) {

    if (!state.initialized) {
        throw new Error(
            "AI models are not ready yet."
        );
    }

    if (!Array.isArray(values) || values.length !== 9) {
        throw new Error(
            "Simulation requires exactly 9 environmental values."
        );
    }

    const invalidIndex =
        values.findIndex(
            value =>
                !Number.isFinite(Number(value))
        );

    if (invalidIndex !== -1) {
        throw new Error(
            `Invalid simulation value for ${CONFIG.FEATURES[invalidIndex]}.`
        );
    }


    /* --------------------------------------------------------
       CREATE MODEL INPUT
       -------------------------------------------------------- */

    const tensor =
        new ort.Tensor(
            "float32",
            Float32Array.from(
                values.map(Number)
            ),
            [1, 9]
        );


    /* --------------------------------------------------------
       CLASSIFIER
       -------------------------------------------------------- */

    const classifier =
        state.models.classifier;

    const classifierInput =
        classifier.inputNames[0];

    const classifierOutputs =
        await classifier.run({
            [classifierInput]: tensor
        });


    const labelOutput =
        classifierOutputs[
            classifier.outputNames[0]
        ];

    const probabilityOutput =
        classifierOutputs[
            classifier.outputNames[1]
        ];


    if (
        !labelOutput ||
        !labelOutput.data
    ) {
        throw new Error(
            "Simulation classifier label output is invalid."
        );
    }

    if (
        !probabilityOutput ||
        !probabilityOutput.data
    ) {
        throw new Error(
            "Simulation classifier probability output is invalid."
        );
    }


    const classification =
        Number(
            labelOutput.data[0]
        );


    const probabilityData =
        Array.from(
            probabilityOutput.data
        );


    if (probabilityData.length < 5) {
        throw new Error(
            "Simulation classifier returned fewer than 5 probabilities."
        );
    }


    const probabilities = {
        0: Number(probabilityData[0]),
        1: Number(probabilityData[1]),
        2: Number(probabilityData[2]),
        3: Number(probabilityData[3]),
        4: Number(probabilityData[4])
    };


    const confidence =
        Math.max(
            probabilities[0],
            probabilities[1],
            probabilities[2],
            probabilities[3],
            probabilities[4]
        );


    /* --------------------------------------------------------
       REGRESSOR
       -------------------------------------------------------- */

    const regressor =
        state.models.regressor;

    const regressorInput =
        regressor.inputNames[0];

    const regressorOutputs =
        await regressor.run({
            [regressorInput]: tensor
        });


    const regressionOutput =
        regressorOutputs[
            regressor.outputNames[0]
        ];


    if (
        !regressionOutput ||
        !regressionOutput.data
    ) {
        throw new Error(
            "Simulation regressor output is invalid."
        );
    }


    let regression =
        Number(
            regressionOutput.data[0]
        );


    /*
     * Same display protection used by the
     * normal AirSafe prediction pipeline.
     */
    regression =
        Math.max(
            0,
            Math.min(
                100,
                regression
            )
        );


    /* --------------------------------------------------------
       ANOMALY DETECTOR
       -------------------------------------------------------- */

    const anomalyModel =
        state.models.anomaly;

    const anomalyInput =
        anomalyModel.inputNames[0];

    const anomalyOutputs =
        await anomalyModel.run({
            [anomalyInput]: tensor
        });


    const anomalyLabelOutput =
        anomalyOutputs[
            anomalyModel.outputNames[0]
        ];

    const anomalyScoreOutput =
        anomalyOutputs[
            anomalyModel.outputNames[1]
        ];


    if (
        !anomalyLabelOutput ||
        !anomalyLabelOutput.data
    ) {
        throw new Error(
            "Simulation anomaly label output is invalid."
        );
    }

    if (
        !anomalyScoreOutput ||
        !anomalyScoreOutput.data
    ) {
        throw new Error(
            "Simulation anomaly score output is invalid."
        );
    }


    const anomalyLabel =
        Number(
            anomalyLabelOutput.data[0]
        );

    const anomalyScore =
        Number(
            anomalyScoreOutput.data[0]
        );


    const anomaly =
        anomalyLabel === -1
            ? "Anomaly"
            : "Normal";


    /* --------------------------------------------------------
       RETURN SIMULATION RESULT
       -------------------------------------------------------- */

    return {
        classification,
        probabilities,
        confidence,
        regression,
        anomaly,
        anomalyScore
    };
}
/* ============================================================
   CLASSIFIER
   ============================================================ */

async function runClassifier(tensor) {

    const session =
        state.models.classifier;

    if (!session) {
        throw new Error(
            "Classifier model is not loaded."
        );
    }

    const inputName =
        session.inputNames[0];

    const outputs =
        await session.run({
            [inputName]: tensor
        });

    console.log(
        "Classifier raw outputs:",
        outputs
    );


    /* --------------------------------------------------------
       OUTPUT 1 — CLASS LABEL
       -------------------------------------------------------- */

    const labelOutput =
        outputs[session.outputNames[0]];

    if (!labelOutput || !labelOutput.data) {
        throw new Error(
            "Classifier label output is missing or invalid."
        );
    }

    const label =
        Number(labelOutput.data[0]);

    state.predictions.classification =
        label;


    /* --------------------------------------------------------
       OUTPUT 2 — CLASS PROBABILITIES
       
       New ONNX model:
       
       label:
           tensor(int64)

       probabilities:
           tensor(float) [batch, 5]
       -------------------------------------------------------- */

    const probabilityOutput =
        outputs[session.outputNames[1]];

    if (
        !probabilityOutput ||
        !probabilityOutput.data
    ) {
        throw new Error(
            "Classifier probability output is missing or invalid."
        );
    }

    const probabilityData =
        Array.from(
            probabilityOutput.data
        );


    if (probabilityData.length < 5) {
        throw new Error(
            `Classifier returned ${probabilityData.length} probabilities. Expected 5.`
        );
    }


    /*
     * Batch size is 1, so the first five
     * values correspond to classes 0–4.
     */

    const probabilities = {
        0: Number(probabilityData[0]),
        1: Number(probabilityData[1]),
        2: Number(probabilityData[2]),
        3: Number(probabilityData[3]),
        4: Number(probabilityData[4])
    };


    state.predictions.probabilities =
        probabilities;


    /* --------------------------------------------------------
       CONFIDENCE
       -------------------------------------------------------- */

    const confidence =
        Math.max(
            probabilities[0],
            probabilities[1],
            probabilities[2],
            probabilities[3],
            probabilities[4]
        );

    state.predictions.confidence =
        confidence;


    console.log(
        "Classifier prediction:",
        label,
        CONFIG.CLASS_LABELS[label]
    );

    console.log(
        "Classifier probabilities:",
        probabilities
    );

    console.log(
        "Classifier confidence:",
        `${(
            confidence * 100
        ).toFixed(2)}%`
    );
}
/* ============================================================
   REGRESSOR
   ============================================================ */

async function runRegressor(tensor) {

    const session =
        state.models.regressor;


    if (!session) {

        throw new Error(
            "Regressor model is not loaded."
        );
    }


    const inputName =
        session.inputNames[0];


    const outputs =
        await session.run({

            [inputName]:
                tensor
        });


    const output =
        outputs[
            session.outputNames[0]
        ];


    let value =
        Number(
            output.data[0]
        );


    /*
     * Keep the displayed score inside the
     * logical 0-100 range.
     */

    value =
        Math.max(
            0,
            Math.min(
                100,
                value
            )
        );


    state.predictions.regression =
        value;


    console.log(
        "Regressor:",
        value
    );
}


/* ============================================================
   ANOMALY DETECTOR
   ============================================================ */

async function runAnomalyDetector(tensor) {

    const session =
        state.models.anomaly;


    if (!session) {

        throw new Error(
            "Anomaly detector is not loaded."
        );
    }


    const inputName =
        session.inputNames[0];


    const outputs =
        await session.run({

            [inputName]:
                tensor
        });


    console.log(
        "Anomaly raw outputs:",
        outputs
    );


    /*
     * Our model has:
     *
     * label
     * scores
     */

    const labelOutput =
        outputs[
            session.outputNames[0]
        ];

    const scoreOutput =
        outputs[
            session.outputNames[1]
        ];


    const label =
        Number(
            labelOutput.data[0]
        );


    const score =
        Number(
            scoreOutput.data[0]
        );


    /*
     * Isolation Forest:
     *
     *  1  = normal
     * -1  = anomaly
     */

    state.predictions.anomaly =
        label === -1
            ? "Anomaly"
            : "Normal";


    state.predictions.anomalyScore =
        score;


    console.log(
        "Anomaly:",
        state.predictions.anomaly
    );

    console.log(
        "Anomaly score:",
        score
    );
}


/* ============================================================
   LOCATION UI
   ============================================================ */

function updateLocationUI() {

    const location =
        state.location;


    setText(
        "locationName",
        location.name
    );


    const secondary =
        [
            location.admin1,
            location.country
        ]
        .filter(Boolean)
        .join(", ");


    setText(
        "locationDetails",
        secondary
    );


    setText(
        "latitudeValue",
        location.latitude.toFixed(4)
    );


    setText(
        "longitudeValue",
        location.longitude.toFixed(4)
    );
}


/* ============================================================
   ENVIRONMENT UI
   ============================================================ */

function updateEnvironmentUI() {

    const data =
        state.environment;


    setText(
        "aqiValue",
        formatNumber(data.AQI)
    );


    setText(
        "pm25Value",
        formatNumber(data.PM2_5)
    );


    setText(
        "pm10Value",
        formatNumber(data.PM10)
    );


    setText(
        "no2Value",
        formatNumber(data.NO2)
    );


    setText(
        "o3Value",
        formatNumber(data.O3)
    );


    setText(
        "so2Value",
        formatNumber(data.SO2)
    );


    setText(
        "temperatureValue",
        formatNumber(data.Temperature)
    );


    setText(
        "humidityValue",
        formatNumber(data.Humidity)
    );


    setText(
        "windValue",
        formatNumber(data.WindSpeed)
    );
}


function formatNumber(value, decimals = 1) {

    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(Number(value))
    ) {

        return "—";
    }


    return Number(value).toFixed(decimals);
}


/* ============================================================
   PREDICTION UI
   ============================================================ */

function updatePredictionUI() {

    const {
        classification,
        probabilities,
        regression,
        anomaly,
        anomalyScore
    } = state.predictions;


    /*
    * Classification
    *
    * Internal model still has classes 0–4.
    * The website presents only 0–2 as the
    * primary Safety Classification.
    *
    * Classes 3 and 4 are grouped visually as 3+
    * without changing the model's actual prediction.
    */
    if (classification !== null) {

        let displayLabel;
        let displayClass;

        if (classification === 0) {
            displayLabel = "Very Low";
            displayClass = "Class 0";
        }
        else if (classification === 1) {
            displayLabel = "Low";
            displayClass = "Class 1";
        }
        else if (classification === 2) {
            displayLabel = "Moderate";
            displayClass = "Class 2";
        }
        else {
            displayLabel = "3+ — Outside primary safety range";
            displayClass = "Class 3+";
        }

        setText(
            "classificationValue",
            displayLabel
        );

        setText(
            "classificationNumber",
            displayClass
        );


        /*
        * Confidence remains the actual model confidence.
        * All five internal classes are still considered.
        */
        const probabilityValues =
            Object.values(probabilities);

        if (probabilityValues.length > 0) {

            const confidence =
                Math.max(
                    ...probabilityValues
                );

            setText(
                "classificationConfidence",
                `${(
                    confidence * 100
                ).toFixed(1)}%`
            );
        }
    }


    /*
     * Regression
     */

    if (regression !== null) {

        setText(
            "healthImpactScore",
            regression.toFixed(1)
        );
    }


    /*
     * Anomaly
     */

    if (anomaly !== null) {

        setText(
            "anomalyValue",
            anomaly
        );


        setText(
            "anomalyScore",
            anomalyScore !== null
                ? anomalyScore.toFixed(4)
                : "—"
        );
    }


    /*
     * Probability bars.
     */

    updateProbabilityBars(
        probabilities
    );
}


/* ============================================================
   PROBABILITY BARS
   ============================================================ */

function updateProbabilityBars(probabilities) {

    /*
     * Only display classes 0–2.
     *
     * Classes 3 and 4 remain inside the model
     * and inside the confidence calculation,
     * but are not shown as normal Safety Classes.
     */

    for (let i = 0; i <= 2; i++) {

        const probability =
            Number(
                probabilities[i] || 0
            );

        const fill =
            $(
                `probability${i}`
            );

        const value =
            $(
                `probabilityValue${i}`
            );

        if (fill) {
            fill.style.width =
                `${(
                    probability * 100
                ).toFixed(1)}%`;
        }

        if (value) {
            value.textContent =
                `${(
                    probability * 100
                ).toFixed(1)}%`;
        }
    }


    /*
     * Hide probability bars for classes 3 and 4
     * if those elements exist in the HTML.
     */
    for (let i = 3; i <= 4; i++) {

        const fill =
            $(
                `probability${i}`
            );

        const value =
            $(
                `probabilityValue${i}`
            );

        if (fill) {
            fill.style.width = "0%";
        }

        if (value) {
            value.textContent = "—";
        }
    }
}

/* ============================================================
   PERSONALISATION
   ============================================================ */

/*
 * Personalisation is an application-level calculation.
 *
 * IMPORTANT:
 * The trained ML models are NOT modified.
 *
 * The personalised index starts from the existing
 * Health Impact Score and applies transparent exposure
 * adjustments based on the user's selected profile.
 *
 * This is NOT a medical risk probability.
 */


/* ------------------------------------------------------------
   PERSONALISATION STATE
   ------------------------------------------------------------ */

state.personalisation = {

    age: "adult",

    activity: "normal",

    exposure: "medium",

    sensitivity: "normal",

    score: null,

    status: null

};


/* ------------------------------------------------------------
   EVENT LISTENER
   ------------------------------------------------------------ */

function attachPersonalisationListeners() {

    const button =
        $("personaliseButton");

    if (!button) {
        return;
    }

    button.addEventListener(
        "click",
        calculatePersonalisedAssessment
    );

}


/* ------------------------------------------------------------
   CALCULATION
   ------------------------------------------------------------ */

function calculatePersonalisedAssessment() {

    /*
     * We need a real environmental analysis first.
     */

    if (
        !state.dataLoaded ||
        state.predictions.regression === null
    ) {

        setText(
            "personalisedStatus",
            "Analyse a location first"
        );

        setText(
            "personalisedExplanation",
            "Search for a location so AirSafe can obtain live environmental data and run its AI models before calculating your personalised assessment."
        );

        return;
    }


    /*
     * Read profile.
     */

    const age =
        $("personalAge")?.value || "adult";

    const activity =
        $("personalActivity")?.value || "normal";

    const exposure =
        $("personalExposure")?.value || "medium";

    const sensitivity =
        $("personalSensitivity")?.value || "normal";


    state.personalisation.age =
        age;

    state.personalisation.activity =
        activity;

    state.personalisation.exposure =
        exposure;

    state.personalisation.sensitivity =
        sensitivity;


    /*
     * Start from the ACTUAL ML regression result.
     */

    const baseScore =
        Number(
            state.predictions.regression
        );


    /*
     * Exposure adjustments.
     *
     * These are deliberately modest because the
     * environmental ML score should remain the
     * dominant component.
     */

    let multiplier = 1.0;


    /* Age */

    if (age === "young") {

        multiplier *= 1.05;

    }
    else if (age === "senior") {

        multiplier *= 1.05;

    }


    /* Activity */

    if (activity === "low") {

        multiplier *= 0.95;

    }
    else if (activity === "high") {

        multiplier *= 1.10;

    }


    /* Time outdoors */

    if (exposure === "short") {

        multiplier *= 0.95;

    }
    else if (exposure === "long") {

        multiplier *= 1.10;

    }


    /* Sensitivity */

    if (sensitivity === "sensitive") {

        multiplier *= 1.08;

    }
    else if (sensitivity === "high") {

        multiplier *= 1.15;

    }


    /*
     * Calculate final personalised index.
     */

    let personalisedScore =
        baseScore * multiplier;


    /*
     * Keep the display within 0–100.
     */

    personalisedScore =
        Math.max(
            0,
            Math.min(
                100,
                personalisedScore
            )
        );


    state.personalisation.score =
        personalisedScore;


    /*
     * Determine a simple application-level
     * interpretation.
     */

    let status;

    if (personalisedScore < 25) {

        status = "Lower exposure impact";

    }
    else if (personalisedScore < 50) {

        status = "Moderate exposure impact";

    }
    else if (personalisedScore < 75) {

        status = "Elevated exposure impact";

    }
    else {

        status = "High exposure impact";

    }


    state.personalisation.status =
        status;


    /*
     * Update UI.
     */

    setText(
        "personalisedScore",
        personalisedScore.toFixed(1)
    );


    setText(
        "personalisedStatus",
        status
    );


    const fill =
        $("personalisedScoreFill");

    if (fill) {

        fill.style.width =
            `${personalisedScore.toFixed(1)}%`;

    }


    /*
     * Generate a transparent explanation.
     */

    const adjustmentPercent =
        ((multiplier - 1) * 100);


    let adjustmentText;

    if (adjustmentPercent > 0.5) {

        adjustmentText =
            `Your selected profile increases the environmental exposure estimate by approximately ${adjustmentPercent.toFixed(0)}% compared with the base AI score.`;

    }
    else if (adjustmentPercent < -0.5) {

        adjustmentText =
            `Your selected profile reduces the environmental exposure estimate by approximately ${Math.abs(adjustmentPercent).toFixed(0)}% compared with the base AI score.`;

    }
    else {

        adjustmentText =
            "Your selected profile produces very little adjustment to the base environmental score.";

    }


    setText(
        "personalisedExplanation",

        `Base Health Impact Score: ${baseScore.toFixed(1)}. ${adjustmentText}`
    );


    console.log(
        "Personalised assessment:",
        {
            baseScore,
            multiplier,
            personalisedScore,
            status,
            profile: {
                age,
                activity,
                exposure,
                sensitivity
            }
        }
    );

}
/* ============================================================
   SIMULATION UI
   ============================================================ */

const SIMULATION_FEATURES = [
    {
        key: "AQI",
        input: "sim-aqi",
        value: "sim-aqi-value"
    },
    {
        key: "PM2_5",
        input: "sim-pm25",
        value: "sim-pm25-value"
    },
    {
        key: "PM10",
        input: "sim-pm10",
        value: "sim-pm10-value"
    },
    {
        key: "NO2",
        input: "sim-no2",
        value: "sim-no2-value"
    },
    {
        key: "SO2",
        input: "sim-so2",
        value: "sim-so2-value"
    },
    {
        key: "O3",
        input: "sim-o3",
        value: "sim-o3-value"
    },
    {
        key: "Temperature",
        input: "sim-temperature",
        value: "sim-temperature-value"
    },
    {
        key: "Humidity",
        input: "sim-humidity",
        value: "sim-humidity-value"
    },
    {
        key: "WindSpeed",
        input: "sim-windspeed",
        value: "sim-windspeed-value"
    }
];


/* ------------------------------------------------------------
   UPDATE SLIDER DISPLAY
   ------------------------------------------------------------ */

function updateSimulationSliderDisplays() {

    SIMULATION_FEATURES.forEach(
        feature => {

            const input =
                $(feature.input);

            const display =
                $(feature.value);

            if (!input || !display) {
                return;
            }

            display.textContent =
                Number(input.value)
                    .toFixed(
                        input.step &&
                        Number(input.step) < 1
                            ? 1
                            : 0
                    );
        }
    );
}


/* ------------------------------------------------------------
   LOAD CURRENT ENVIRONMENT INTO SLIDERS
   ------------------------------------------------------------ */

function loadCurrentConditionsIntoSimulation() {

    if (!state.dataLoaded) {
        return;
    }

    SIMULATION_FEATURES.forEach(
        feature => {

            const input =
                $(feature.input);

            if (!input) {
                return;
            }

            const currentValue =
                Number(
                    state.environment[
                        feature.key
                    ]
                );

            if (
                Number.isFinite(
                    currentValue
                )
            ) {

                /*
                 * Clamp to slider range so unusual
                 * environmental values don't break
                 * the HTML range input.
                 */
                const min =
                    Number(input.min);

                const max =
                    Number(input.max);

                const safeValue =
                    Math.max(
                        min,
                        Math.min(
                            max,
                            currentValue
                        )
                    );

                input.value =
                    safeValue;
            }
        }
    );

    updateSimulationSliderDisplays();
}


/* ------------------------------------------------------------
   READ SIMULATION VALUES
   ------------------------------------------------------------ */

function getSimulationValues() {

    return SIMULATION_FEATURES.map(
        feature => {

            const input =
                $(feature.input);

            if (!input) {
                throw new Error(
                    `Simulation input missing: ${feature.input}`
                );
            }

            return Number(
                input.value
            );
        }
    );
}


/* ------------------------------------------------------------
   DISPLAY CLASSIFICATION
   ------------------------------------------------------------ */

function getSimulationClassificationLabel(
    classification
) {

    if (classification === 0) {
        return "Very Low";
    }

    if (classification === 1) {
        return "Low";
    }

    if (classification === 2) {
        return "Moderate";
    }

    /*
     * Internal classes 3 and 4 remain untouched.
     * The website groups them as 3+.
     */
    return "3+ — Outside primary safety range";
}


/* ------------------------------------------------------------
   DISPLAY CURRENT RESULTS
   ------------------------------------------------------------ */

function updateSimulationCurrentResults() {

    if (!state.dataLoaded) {
        return;
    }


    setText(
        "sim-current-class",
        getSimulationClassificationLabel(
            state.predictions.classification
        )
    );


    setText(
        "sim-current-confidence",
        state.predictions.confidence !== null
            ? `${(
                state.predictions.confidence * 100
            ).toFixed(1)}%`
            : "—"
    );


    setText(
        "sim-current-score",
        state.predictions.regression !== null
            ? state.predictions.regression.toFixed(1)
            : "—"
    );


    setText(
        "sim-current-anomaly",
        state.predictions.anomaly !== null
            ? state.predictions.anomaly
            : "—"
    );
}


/* ------------------------------------------------------------
   DISPLAY SIMULATION RESULTS
   ------------------------------------------------------------ */

function updateSimulationResults(
    result
) {

    setText(
        "sim-result-class",
        getSimulationClassificationLabel(
            result.classification
        )
    );


    setText(
        "sim-result-confidence",
        `${(
            result.confidence * 100
        ).toFixed(1)}%`
    );


    setText(
        "sim-result-score",
        result.regression.toFixed(1)
    );


    setText(
        "sim-result-anomaly",
        result.anomaly
    );


    /*
     * Compare simulated score with
     * the current live score.
     */

    const currentScore =
        Number(
            state.predictions.regression
        );

    const simulatedScore =
        Number(
            result.regression
        );


    const difference =
        simulatedScore -
        currentScore;


    let impactText;


    if (
        !Number.isFinite(
            difference
        )
    ) {

        impactText =
            "Simulation completed successfully.";
    }

    else if (
        Math.abs(difference) < 0.05
    ) {

        impactText =
            "The simulated conditions produce almost the same Health Impact Score as the current conditions.";
    }

    else if (
        difference < 0
    ) {

        impactText =
            `The simulated conditions reduce the Health Impact Score by ${Math.abs(difference).toFixed(1)} points compared with the current conditions.`;
    }

    else {

        impactText =
            `The simulated conditions increase the Health Impact Score by ${difference.toFixed(1)} points compared with the current conditions.`;
    }


    setText(
        "simulation-impact-text",
        impactText
    );
}


/* ------------------------------------------------------------
   RUN SIMULATION
   ------------------------------------------------------------ */

async function runSimulation() {

    const button =
        $("run-simulation-btn");

    const status =
        $("simulation-status");


    if (!state.initialized) {

        if (status) {
            status.textContent =
                "AI models not ready";
        }

        return;
    }


    if (!state.dataLoaded) {

        if (status) {
            status.textContent =
                "Search for a location first";
        }

        setText(
            "simulation-impact-text",
            "Please analyse a location first so the simulation has real environmental conditions to work from."
        );

        return;
    }


    try {

        if (button) {
            button.disabled = true;
            button.classList.add("loading");
        }


        if (status) {
            status.textContent =
                "Running AI models...";
        }


        const values =
            getSimulationValues();


        console.log(
            "============================================================"
        );

        console.log(
            "AIRSAFE WHAT-IF SIMULATION"
        );

        console.table(
            CONFIG.FEATURES.map(
                (feature, index) => ({
                    Feature: feature,
                    Value: values[index]
                })
            )
        );


        const result =
            await runSimulationModels(
                values
            );


        console.log(
            "Simulation result:",
            result
        );


        updateSimulationResults(
            result
        );


        if (status) {
            status.textContent =
                "Simulation complete";
        }


    } catch (error) {

        console.error(
            "Simulation failed:",
            error
        );


        if (status) {
            status.textContent =
                "Simulation failed";
        }


        setText(
            "simulation-impact-text",
            error.message ||
            "Unable to run the simulation."
        );


    } finally {

        if (button) {
            button.disabled = false;
            button.classList.remove("loading");
        }
    }
}


/* ------------------------------------------------------------
   RESET SIMULATION
   ------------------------------------------------------------ */

function resetSimulation() {

    if (!state.dataLoaded) {

        setText(
            "simulation-impact-text",
            "Search for a location first to load the current environmental conditions."
        );

        return;
    }


    loadCurrentConditionsIntoSimulation();


    /*
     * Reset result area to the live prediction.
     */

    updateSimulationCurrentResults();


    setText(
        "sim-result-class",
        "—"
    );

    setText(
        "sim-result-confidence",
        "—"
    );

    setText(
        "sim-result-score",
        "—"
    );

    setText(
        "sim-result-anomaly",
        "—"
    );


    setText(
        "simulation-impact-text",
        "Simulation reset to the current environmental conditions. Adjust the sliders and run the AI simulation."
    );


    const status =
        $("simulation-status");

    if (status) {
        status.textContent =
            "Ready to simulate";
    }
}
/* ============================================================
   DEBUGGING
   ============================================================ */

window.AirSafe =
    {

        state,

        config:
            CONFIG,

        runModels:
            runAllModels,

        fetchEnvironmentalData,

        geocodeLocation
    };


console.log(
    "AirSafe debug interface available as window.AirSafe"
);
/* ============================================================
   FEATURE IMPORTANCE
   ============================================================ */

async function loadFeatureImportance() {

    const canvas = $("featureImportanceChart");
    const emptyState = $("featureImportanceEmpty");

    if (!canvas || !emptyState) {
        console.warn(
            "Feature importance elements not found."
        );
        return;
    }

    try {

        const response =
            await fetch("metadata.json");

        if (!response.ok) {
            throw new Error(
                `metadata.json returned ${response.status}`
            );
        }

        const metadata =
            await response.json();

        const importance =
            metadata.classifier_feature_importance;

        if (
            !Array.isArray(importance) ||
            importance.length === 0
        ) {
            throw new Error(
                "No classifier feature importance found."
            );
        }

        drawFeatureImportance(
            canvas,
            emptyState,
            importance
        );

    } catch (error) {

        console.error(
            "Feature importance loading failed:",
            error
        );

        emptyState.textContent =
            "Feature importance unavailable.";

        emptyState.style.display =
            "flex";
    }
}
function drawFeatureImportance(
    canvas,
    emptyState,
    importance
) {

    const ctx =
        canvas.getContext("2d");

    const textColor = "#e8edf5";
    const mutedColor = "#94a3b8";
    const accentColor = "#62d6c7";

    if (!ctx) {
        return;
    }

    const rect =
        canvas.getBoundingClientRect();

    const width =
        Math.max(
            500,
            Math.floor(rect.width)
        );

    const height =
        Math.max(
            320,
            Math.floor(rect.height)
        );

    const dpr =
        window.devicePixelRatio || 1;

    canvas.width =
        width * dpr;

    canvas.height =
        height * dpr;

    canvas.style.width =
        `${width}px`;

    canvas.style.height =
        `${height}px`;

    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );

    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    /*
     * Sort by absolute importance so the most
     * influential features appear first.
     */

    const sorted =
        [...importance]
            .sort(
                (a, b) =>
                    Math.abs(b.importance) -
                    Math.abs(a.importance)
            );


    const maxImportance =
        Math.max(
            ...sorted.map(
                item =>
                    Math.abs(
                        Number(item.importance)
                    )
            ),
            0.001
        );


    const left =
        105;

    const right =
        25;

    const top =
        25;

    const bottom =
        30;

    const chartWidth =
        width - left - right;

    const chartHeight =
        height - top - bottom;

    const rowHeight =
        chartHeight / sorted.length;


    ctx.font =
        "13px Arial";

    ctx.textBaseline =
        "middle";


    sorted.forEach(
        (item, index) => {

            const feature =
                item.feature;

            const value =
                Number(item.importance);

            const absValue =
                Math.abs(value);

            const y =
                top +
                rowHeight * index +
                rowHeight / 2;

            const barWidth =
                (absValue / maxImportance) *
                chartWidth;

            /*
             * Feature name
             */
            ctx.fillStyle = textColor;
            ctx.textAlign =
                "right";

            ctx.fillText(
                feature,
                left - 12,
                y
            );


            /*
             * Baseline
             */
            ctx.fillStyle = "rgba(255,255,255,0.08)";
            ctx.fillRect(
                left,
                y - 5,
                chartWidth,
                10
            );


            /*
             * Importance bar
             */
            ctx.fillStyle = accentColor;
            ctx.fillRect(
                left,
                y - 6,
                barWidth,
                12
            );


            /*
             * Numerical value
             */
            ctx.fillStyle = mutedColor;
            ctx.textAlign =
                "left";

            ctx.fillText(
                value.toFixed(3),
                left + barWidth + 8,
                y
            );
        }
    );


    /*
     * Hide the "waiting" message once
     * the chart has successfully rendered.
     */

    emptyState.style.display =
        "none";
}
/* ============================================================
   MAP
   ============================================================ */

let airSafeMap = null;
let airSafeMarker = null;
let airSafeAccuracyCircle = null;


function initializeMap() {

    const mapElement =
        $("map");

    if (!mapElement) {
        console.warn(
            "AirSafe map container not found."
        );
        return;
    }

    /*
     * Prevent initializing Leaflet more than once.
     */

    if (airSafeMap) {
        return;
    }


    /*
     * Default view.
     * This is only temporary until a location is selected.
     */

    airSafeMap =
        L.map(
            mapElement,
            {
                zoomControl: true,
                attributionControl: true
            }
        )
        .setView(
            [20, 0],
            2
        );


    /*
     * CARTO Voyager basemap
     */

    L.tileLayer(
        CONFIG.MAP.TILE_URL +
        "?key=" +
        encodeURIComponent(
            CONFIG.MAP.CARTO_API_KEY
        ),
        {
            subdomains:
                ["a", "b", "c", "d"],

            maxZoom: 20,

            attribution:
                '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> ' +
                '&copy; <a href="https://carto.com/attributions" target="_blank">CARTO</a>'
        }
    ).addTo(
        airSafeMap
    );


    console.log(
        "AirSafe map initialized."
    );
}
function updateMapLocation() {

    if (
        !airSafeMap ||
        !state.location
    ) {
        return;
    }

    const lat =
        Number(
            state.location.latitude
        );

    const lon =
        Number(
            state.location.longitude
        );

    if (
        !Number.isFinite(lat) ||
        !Number.isFinite(lon)
    ) {
        return;
    }


    /*
     * Remove previous marker.
     */

    if (airSafeMarker) {

        airSafeMap.removeLayer(
            airSafeMarker
        );

        airSafeMarker = null;
    }


    /*
     * Remove previous circle.
     */

    if (airSafeAccuracyCircle) {

        airSafeMap.removeLayer(
            airSafeAccuracyCircle
        );

        airSafeAccuracyCircle = null;
    }


    /*
     * Marker
     */

    airSafeMarker =
        L.marker(
            [lat, lon]
        )
        .addTo(
            airSafeMap
        );


    /*
     * Location popup
     */

    const locationName =
        state.location.name ||
        "Selected location";

    airSafeMarker.bindPopup(
        `<strong>${locationName}</strong><br>` +
        `${lat.toFixed(4)}, ` +
        `${lon.toFixed(4)}`
    );


    /*
     * Center map on selected location.
     */

    airSafeMap.setView(
        [lat, lon],
        11,
        {
            animate: true
        }
    );


    /*
     * Leaflet sometimes calculates the container
     * dimensions incorrectly when the map is inside
     * a dynamically rendered section.
     */

    setTimeout(
        () => {
            airSafeMap.invalidateSize();
        },
        200
    );


    console.log(
        "Map location updated:",
        {
            name: locationName,
            latitude: lat,
            longitude: lon
        }
    );
}
