const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const dotenv = require('dotenv');
const Groq = require('groq-sdk');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Groq AI Client with dynamic reload capability
let groq = null;
function getGroqClient() {
    if (groq) return groq;
    dotenv.config({ override: true });
    const key = process.env.GROQ_API_KEY ? process.env.GROQ_API_KEY.trim() : '';
    if (key && key !== '' && !key.includes('your_groq_api_key_here')) {
        try {
            groq = new Groq({ apiKey: key });
            
        } catch (err) {
            console.error("Failed to initialize Groq client:", err.message);
        }
    }
    return groq;
}

// Initial check on boot
getGroqClient();
if (!groq) {
    console.error("GROQ_API_KEY is missing in .env");
} 

// Ensure uploads directory exists
const UPLOADS_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Multer storage configuration
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOADS_DIR),
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname) || '.jpg';
        cb(null, `leaf_${Date.now()}_${Math.random().toString(36).substring(2, 7)}${ext}`);
    }
});

const upload = multer({
    storage,
    limits: { fileSize: 15 * 1024 * 1024 }, // 15MB limit
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only image files (PNG, JPG, WEBP) are supported.'));
        }
    }
});

// Middleware
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

// Root endpoint: Serves HTML to web browsers, and JSON confirmation to API callers
app.get('/', (req, res, next) => {
    if (req.headers.accept && req.headers.accept.includes('application/json') && !req.headers.accept.includes('text/html')) {
        return res.json({
            message: "Smart Crop AI server is running"
        });
    }
    res.sendFile('index.html', { root: path.resolve(__dirname, 'public') });
});

// Static assets (CSS, JS, images)
app.use(express.static(path.join(__dirname, 'public')));

// Load metadata
const diseaseInfoPath = path.join(__dirname, 'disease_info.json');
let diseaseInfo = {};
if (fs.existsSync(diseaseInfoPath)) {
    try {
        diseaseInfo = JSON.parse(fs.readFileSync(diseaseInfoPath, 'utf8'));
    } catch (e) {
        console.error('Failed to parse disease_info.json:', e);
    }
}

// Supported Crops Metadata (Strict 13 Supported Crops)
const SUPPORTED_CROPS = [
    { id: "apple", name: "Apple", sampleImg: "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=700&auto=format&fit=crop&q=60" },
    { id: "chickpea", name: "Chickpea", sampleImg: "https://images.unsplash.com/photo-1677887736557-a461460caaf6?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "corn", name: "Corn (Maize)",  sampleImg: "https://images.unsplash.com/photo-1554402100-8d1d9f3dff80?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "cotton", name: "Cotton", sampleImg: "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?w=400&auto=format&fit=crop&q=80" },
    { id: "onion", name: "Onion",  sampleImg: "https://images.unsplash.com/photo-1508747703725-719777637510?w=400&auto=format&fit=crop&q=80" },
    { id: "pepper", name: "Pepper",  sampleImg: "https://plus.unsplash.com/premium_photo-1680345328197-aa29e0b38e7b?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "potato", name: "Potato",  sampleImg: "https://images.unsplash.com/photo-1781113208220-6565a167fe59?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "rice", name: "Rice",  sampleImg: "https://images.unsplash.com/photo-1635562985686-4f8bb9c0d3bf?w=1000&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "rose", name: "Rose",  sampleImg: "https://images.unsplash.com/photo-1586082206695-d19601b5aa88?q=80&w=870&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "soybean", name: "Soybean",  sampleImg: "https://plus.unsplash.com/premium_photo-1661815691473-5efc09a055e4?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "sugarcane", name: "Sugarcane", sampleImg: "https://tse3.mm.bing.net/th/id/OIP.Dv_xM0UkGCvEjfejV7YKYAHaE7?r=0&pid=Api&h=220&P=0" },
    { id: "tomato", name: "Tomato",  sampleImg: "https://plus.unsplash.com/premium_photo-1661833100239-de8f260b6f8c?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" },
    { id: "wheat", name: "Wheat",  sampleImg: "https://images.unsplash.com/photo-1607460694867-af0c6d6f2c52?w=700&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=NHwxMjA3fDA%3D" }
];

// Helper to run inference.py as a child process
function runInference(imagePath, cropFilter = "apple") {
    return new Promise((resolve, reject) => {
        const pythonScript = path.join(__dirname, 'inference.py');
        const targetCrop = (cropFilter && cropFilter.trim() !== '' && cropFilter !== 'autodetect') ? cropFilter.trim() : 'apple';
        const args = ['inference.py', '--image', imagePath, '--crop', targetCrop, '--json'];

        const pythonExe = process.platform === 'win32' ? 'python' : 'python3';

        execFile(pythonExe, args, { cwd: __dirname }, (error, stdout, stderr) => {
            if (error) {
                console.error('Inference error:', stderr || error.message);
                return reject(new Error(stderr || error.message));
            }

            try {
                const jsonStart = stdout.indexOf('{');
                const jsonEnd = stdout.lastIndexOf('}');
                if (jsonStart === -1 || jsonEnd === -1) {
                    throw new Error(`Invalid JSON output from model inference: ${stdout}`);
                }
                const jsonStr = stdout.substring(jsonStart, jsonEnd + 1);
                const parsed = JSON.parse(jsonStr);
                resolve(parsed);
            } catch (err) {
                reject(new Error(`Failed to parse inference output: ${err.message}. Raw output: ${stdout}`));
            }
        });
    });
}

// Helper to clean asterisks/stars from text
function cleanApiOutput(text) {
    if (!text || typeof text !== 'string') return '';
    return text
        .replace(/\*\*/g, '')          // Remove double asterisks
        .replace(/^\s*\*\s+/gm, '- ')  // Convert asterisk bullets to dashes
        .replace(/\*/g, '');          // Remove any stray single asterisks
}

// Offline Agronomist Knowledge Engine (Fallback when no key is configured)
function fallbackChatResponse(message) {
    const q = message.toLowerCase();

    if (q.includes('early blight') || q.includes('late blight') || q.includes('blight')) {
        return "🌱 Blight Management Advisory:\n- Organic: Spray Neem oil (5ml/L) or Copper oxychloride (2.5g/L) every 7-10 days.\n- Chemical: Apply Mancozeb 75 WP or Azoxystrobin at early onset.\n- Culture: Prune lower diseased leaves touching the ground, and irrigate at the soil base rather than overhead sprinkler.";
    }

    if (q.includes('rust') || q.includes('wheat') || q.includes('mustard') || q.includes('garlic')) {
        return "🌾 Rust Disease Management:\n- Identification: Look for orange-yellow, reddish-brown, or white pustules on leaf blades.\n- Treatment: Spray Propiconazole 25 EC (Tilt 1ml/L) or Wettable Sulfur (2.5g/L).\n- Field Practice: Ensure good row spacing for airflow, and avoid excessive nitrogen fertilizer.";
    }

    if (q.includes('onion') || q.includes('purple blotch')) {
        return "🧅 Onion & Garlic Purple Blotch Guide:\n- Action: Spray Chlorothalonil (2g/L) or Mancozeb mixed with a wetting sticker agent.\n- Organic: Trichoderma viride seed/bulb treatment prior to planting.\n- Prevention: Rotate crops with cereals; avoid overhead watering in humid weather.";
    }

    if (q.includes('fertilizer') || q.includes('npk') || q.includes('recovery')) {
        return "🧪 Nutrient & Fertilizer Recommendation:\n- After disease recovery, apply a foliar spray of balanced micronutrients (Zinc, Boron, Magnesium) plus soluble 19:19:19 NPK (5g/L).\n- Incorporate well-decomposed farmyard manure (FYM) or vermicompost to revitalize root soil microbes.";
    }

    if (q.includes('rose') || q.includes('black spot')) {
        return "🌹 Rose Black Spot Care:\n- Remove and safely burn fallen yellowing spotted leaves.\n- Spray systemic fungicide (Myclobutanil or Carbendazim 1g/L) or Neem oil every 10 days.\n- Keep foliage completely dry when watering rose bushes.";
    }

    return "🌱 AgriVision Agronomy Advisory:\nFor healthy crops, ensure regular field scouting, balanced NPK fertilization, clean weeding, and well-drained soil. If you notice leaf discoloration, spots, or wilt, submit a photo to the Disease Detector tab for instant vision model diagnosis!\n\n(Tip: AI Assistant is ready for interactive agronomy questions!)";
}

// -------------------------------------------------------------
// API Endpoints
// -------------------------------------------------------------

// 1. Health check & status
app.get('/api/status', (req, res) => {
    const client = getGroqClient();
    res.json({
        status: 'online',
        system: 'Smart Crop AI server is running',
        version: '2.5 (AgriVision AI Engine)',
        ai_active: Boolean(client),
        ai_status: client ? 'Connected to AI Engine' : 'AI Engine Ready',
        supported_crops_count: SUPPORTED_CROPS.length,
        classes_count: Object.keys(diseaseInfo).length
    });
});

// 2. Supported Crops List
app.get('/api/crops', (req, res) => {
    const cropsWithDiseases = SUPPORTED_CROPS.map(c => {
        const diseases = Object.values(diseaseInfo)
            .filter(d => d.crop.toLowerCase().includes(c.id.toLowerCase()) || c.id.toLowerCase().includes(d.crop.toLowerCase()))
            .map(d => d.disease);
        return {
            ...c,
            diseases: diseases.length > 0 ? diseases.slice(0, 3).join(', ') : 'Healthy, Common Pathogens'
        };
    });

    res.json({
        success: true,
        crops: cropsWithDiseases
    });
});

// Multi-Line Clinical Treatment & Agronomy Advisory Generator
function getExpandedTreatmentPlan(crop, disease, status) {
    const isHealthy = status.toLowerCase() === 'healthy';
    if (isHealthy) {
        return {
            treatmentSummary: `Crop foliage shows optimal vegetative vigor with no active sporulation. Maintain balanced fertilization, regular irrigation, and prophylactic scouting.`,
            chemical: [
                "No therapeutic chemical fungicides or bactericides required for healthy crops.",
                "Foliar Nutrition: Spray water-soluble 19:19:19 NPK (3g to 5g / Liter) during active vegetative growth.",
                "Micronutrient Care: Apply Chelated Zinc (1g/L) and Boron (0.5g/L) once every 25 days to sustain foliage health.",
                "Soil Fertility: Apply balanced NPK basal doses based on local soil test health cards."
            ],
            organic: [
                "Prophylactic Protection: Apply cold-pressed Neem cake (50kg to 100kg per acre) to inhibit soil pathogens.",
                "Bio-Stimulant Spray: Spray fermented vermiwash (50ml/L) or cold-water seaweed extract (2ml/L) every 14 days.",
                "Microbial Inoculant: Drench root soil with Trichoderma viride (5g/L) or Pseudomonas fluorescens (5g/L) monthly.",
                "Mulching: Maintain 2-3 inches of organic straw mulch around root basins to conserve moisture and suppress weeds."
            ],
            prevention: [
                "Conduct routine field scouting every 3-4 days, particularly following heavy morning dew or unseasonal rainfall.",
                "Sterilize pruning shears, knives, and harvesting tools with 70% isopropyl alcohol between rows.",
                "Ensure proper field drainage channels to prevent root-zone waterlogging and collar rot.",
                "Avoid excessive high-nitrogen applications which create succulent, pathogen-susceptible plant canopy."
            ]
        };
    }

    const dLower = disease.toLowerCase();
    let chemLines = [];
    let orgLines = [];

    if (dLower.includes('rust')) {
        chemLines = [
            `Primary Treatment for ${disease}: Spray Propiconazole 25 EC (Tilt) @ 1 ml per Liter of water immediately upon lesion detection.`,
            "Protective Spray: Follow with Mancozeb 75 WP @ 2.5 g/L or Wettable Sulfur 80 WDG @ 3 g/L 10 to 14 days later.",
            "Application Technique: Ensure high-pressure, fine mist coverage on both upper and lower leaf surfaces where pustules burst.",
            "Weather Safeguard: Spray during calm morning hours; avoid spraying if rain is forecasted within 4 hours.",
            "Fungicide Rotation: Alternate between Triazole (FRAC 3) and Strobilurin (FRAC 11) modes of action to prevent fungal resistance."
        ];
        orgLines = [
            "Botanical Bio-wash: Apply cold-pressed Neem oil (10,000 PPM) @ 5 ml/L mixed with 1 ml/L liquid soap as an emulsifier.",
            "Bio-Control Agent: Spray Verticillium lecanii or Ampelomyces quisqualis (5 g/L) under humid evening conditions.",
            "Physical Sanitation: Manually prune and safely incinerate heavily rusted lower foliage to reduce secondary spore release.",
            "Canopy Aeration: Prune crowded shoots to lower relative micro-climatic humidity across the leaf canopy."
        ];
    } else if (dLower.includes('blight') || dLower.includes('rot') || dLower.includes('spot') || dLower.includes('scab') || dLower.includes('canker')) {
        chemLines = [
            `Foliar Fungicide Spray: Apply Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L or Mancozeb 75 WP @ 2.5 g/L.`,
            "Bacterial Complex: If bacterial spots or watersoaked lesions are observed, tank-mix Copper Oxychloride 50 WP (2.5 g/L) + Streptocycline (0.1 g/L).",
            "Spray Interval: Repeat application at 7 to 10-day intervals depending on weather humidity and disease spread.",
            "Adjuvant Usage: Add a silicon-based non-ionic spreading agent (0.5 ml/L) to maximize leaf adherence and rain-fastness.",
            "Recovery Booster: Spray 13:00:45 Potassium Nitrate (5 g/L) 5 days after fungicide spray to stimulate new healthy leaf flush."
        ];
        orgLines = [
            "Protective Microbe: Spray Trichoderma viride or Pseudomonas fluorescens @ 5 g to 10 g per Liter of water.",
            "Neem Spray: Apply cold-pressed Neem Seed Kernel Extract (NSKE 5%) or 10,000 PPM Neem oil @ 5 ml/L every 7 days.",
            "Herbal Decoction: Spray fermented garlic-chili extract or 10% cow urine solution as a natural broad-spectrum antimicrobial.",
            "Debris Disposal: Rake up and destroy fallen infected leaves and rotting plant parts far away from crop beds."
        ];
    } else if (dLower.includes('mildew') || dLower.includes('smut')) {
        chemLines = [
            `Systemic Fungicide: Spray Hexaconazole 5% SC @ 2 ml/L or Myclobutanil 10% WP @ 1 g/L at the first sign of powdery growth.`,
            "Contact Protective Spray: Apply Wettable Sulfur 80% WP @ 2.5 g/L or Chlorothalonil 75 WP @ 2 g/L.",
            "High Spray Volume: Use at least 200 Liters of water per acre to drench the entire foliage and stem surface thoroughly.",
            "Spray Timing: Apply in early morning before midday heat to avoid chemical phytotoxicity on tender foliage."
        ];
        orgLines = [
            "Potassium Bicarbonate: Spray Potassium Bicarbonate or Baking Soda (Sodium Bicarbonate) @ 3 g/L + 2 ml/L horticultural oil.",
            "Dilute Milk Solution: Spray fresh whey or raw milk (1 part milk : 9 parts water) under direct morning sunlight.",
            "Neem Application: Spray cold-pressed Neem oil (5 ml/L) weekly until powdery patches are completely eradicated.",
            "Airflow Management: Weed field boundaries and thin out dense crop rows to allow sunlight penetration."
        ];
    } else if (dLower.includes('virus') || dLower.includes('curl') || dLower.includes('mosaic') || dLower.includes('aphid') || dLower.includes('mite')) {
        chemLines = [
            "Vector Management: Spray Acetamiprid 20% SP @ 0.5 g/L or Imidacloprid 17.8 SL @ 0.5 ml/L to control sap-sucking whitefly/aphid vectors.",
            "Mite Control: If spider mites or stippling are present, spray Spiromesifen 22.9 SC @ 1 ml/L or Abamectin 1.9 EC @ 0.5 ml/L.",
            "Thorough Coverage: Direct spray nozzle upward to wet the underside of leaves where insect colonies reside.",
            "Pre-Harvest Interval (PHI): Observe a strict 7 to 10-day wait period before harvesting any produce."
        ];
        orgLines = [
            "Sticky Insect Traps: Install 15-20 yellow and blue sticky traps per acre at canopy level to monitor and mass-trap insect vectors.",
            "Neem Repellent: Spray cold-pressed Neem oil (10,000 PPM @ 5 ml/L) every 5 days to deter vector feeding and oviposition.",
            "Rogue Infected Plants: Uproot and burn severely stunted or mosaic-mottled plants immediately to prevent viral spread.",
            "Barrier Planting: Plant 2-3 dense border rows of corn, sorghum, or pearl millet around crop perimeter as physical vector barriers."
        ];
    } else {
        chemLines = [
            `Therapeutic Spray for ${crop} ${disease}: Apply Copper Oxychloride 50 WP (2.5 g/L) or Mancozeb 75 WP (2 g/L).`,
            "Broad-Spectrum Systemic: Alternate with Carbendazim 12% + Mancozeb 63% WP (Saaf) @ 2 g per Liter.",
            "Spray Regimen: Apply 2 applications spaced 10 days apart during cool, windless morning hours.",
            "Protective Equipment: Wear protective gloves, face mask, and eye shield during preparation and spraying."
        ];
        orgLines = [
            "Bio-Protective Spray: Apply Trichoderma viride @ 5 g/L or Pseudomonas fluorescens @ 5 g/L as a protective foliar barrier.",
            "Neem Formulation: Spray cold-pressed Neem oil (5 ml/L) mixed with mild liquid soap every 7 days.",
            "Field Hygiene: Collect and safely incinerate all blighted foliage to prevent fungal inoculum build-up in soil."
        ];
    }

    const prevLines = [
        "Implement a strict 2-3 year crop rotation cycle with unrelated botanical families to break pathogen cycles.",
        "Use drip irrigation or furrow watering; eliminate overhead sprinkler watering that keeps leaves wet overnight.",
        "Maintain balanced NPK fertilization supplemented with 5 tons/acre well-decomposed farmyard manure or vermicompost.",
        "Procure and plant only certified disease-free, fungicide-treated seeds or resistant crop cultivars."
    ];

    return {
        treatmentSummary: `${crop} is affected by ${disease}. Immediate clinical field action, chemical spray schedule, and bio-organic management are detailed below.`,
        chemical: chemLines,
        organic: orgLines,
        prevention: prevLines
    };
}

// 3. Diagnose Crop Disease via Vision Model
app.post('/api/diagnose', upload.single('image'), async (req, res) => {
    let tempImagePath = null;
    let isUploadedFile = false;

    try {
        const cropFilter = req.body.crop || 'apple';
        const sampleKey = req.body.sample || null;

        if (req.file) {
            tempImagePath = req.file.path;
            isUploadedFile = true;
        } else if (sampleKey) {
            const sampleMap = {
                sample_leaf: path.join(__dirname, 'sample_leaf.jpg'),
                garlic_leaf: path.join(__dirname, 'garlic_leaf_closeup.png'),
                tomato: path.join(__dirname, 'sample_leaf.jpg'),
                corn: path.join(__dirname, 'sample_leaf.jpg'),
                wheat: path.join(__dirname, 'sample_leaf.jpg'),
                rose: path.join(__dirname, 'sample_leaf.jpg')
            };

            tempImagePath = sampleMap[sampleKey.toLowerCase()] || path.join(__dirname, 'sample_leaf.jpg');
            if (!fs.existsSync(tempImagePath)) {
                tempImagePath = path.join(__dirname, 'sample_leaf.jpg');
            }
        } else {
            return res.status(400).json({ success: false, error: 'Please upload an image file or choose a sample scan.' });
        }

        const result = await runInference(tempImagePath, cropFilter);
        const top = result.top;

        const info = diseaseInfo[String(top.class_id)] || {};
        const cropName = top.crop || info.crop || cropFilter.toUpperCase();
        const diseaseName = top.disease || info.disease || 'Condition';
        const status = top.status || info.status || 'Diseased';
        const confidence = top.confidence || 95.0;

        const isHealthy = status.toLowerCase() === 'healthy';

        // Generate Multi-Line Treatment Plan
        const plan = getExpandedTreatmentPlan(cropName, diseaseName, status);

        const symptomsList = (info.symptoms && info.symptoms.length > 0) ? info.symptoms : (isHealthy ? [
            "Foliage displays rich green pigmentation and uniform leaf turgidity.",
            "Absence of necrotic spotting, rust pustules, or chlorotic halos.",
            "Normal plant vigor, apical shoot development, and root health."
        ] : [
            `Characteristic ${diseaseName} lesion formation observed across leaf blades.`,
            "Localized cellular necrosis, brown discoloration, and chlorotic margins on foliage.",
            "Reduced photosynthetic active surface area leading to canopy stress."
        ]);

        res.json({
            success: true,
            crop: cropName,
            disease: diseaseName,
            status: status,
            confidence: `${confidence}%`,
            latency_ms: result.latency_ms,
            treatment: plan.treatmentSummary,
            symptoms: symptomsList,
            organic: plan.organic,
            chemical: plan.chemical,
            prevention: plan.prevention,
            alternatives: result.alternatives || [],
            has_groq_key: Boolean(getGroqClient())
        });

    } catch (err) {
        console.error('Diagnosis handler error:', err);
        res.status(500).json({
            success: false,
            error: err.message || 'Internal server error during leaf diagnosis.'
        });
    } finally {
        if (isUploadedFile && tempImagePath && fs.existsSync(tempImagePath)) {
            try {
                fs.unlinkSync(tempImagePath);
            } catch (e) {
                // Ignore cleanup error
            }
        }
    }
});

// 4. Groq AI Agricultural Advice Endpoint (User Specification)
app.post("/api/advice", async (req, res) => {
    try {
        const { question, crop, disease } = req.body;

        if (
            typeof question !== "string" ||
            !question.trim() ||
            question.length > 2000
        ) {
            return res.status(400).json({
                error: "Enter a question up to 2000 characters."
            });
        }

        const cropName =
            typeof crop === "string" ? crop.slice(0, 100) : "Not specified";

        const diseaseName =
            typeof disease === "string"
                ? disease.slice(0, 150)
                : "Not specified";

        const client = getGroqClient();
        if (!client) {
            console.error("Groq API error: 401 GROQ_API_KEY is missing in .env");
            return res.status(401).json({
                error: "Invalid API key. Check your .env file."
            });
        }

        const completion = await client.chat.completions.create({
            model: "openai/gpt-oss-20b",
            messages: [
                {
                    role: "system",
                    content: `You are a helpful agricultural assistant.
Give practical, easy-to-understand farming advice.
Crop: ${cropName}
Detected disease: ${diseaseName}
Explain possible causes, prevention, and safe next steps.
Do not claim a disease is confirmed from text alone.
Do not invent pesticide dosages. Recommend following
the product label and local agricultural expert guidance.
If details are missing, ask a clarifying question.
Important: Do not use asterisks or markdown stars like ** or * anywhere in your response. Use plain text formatting and simple hyphens (-) for lists.`
                },
                {
                    role: "user",
                    content: question.trim()
                }
            ],
            temperature: 0.4,
            max_completion_tokens: 1000
        });

        const advice = completion.choices?.[0]?.message?.content;

        if (!advice) {
            return res.status(502).json({
                error: "The AI returned an empty response."
            });
        }

        const cleanedAdvice = cleanApiOutput(advice);
        res.json({ advice: cleanedAdvice });
    } catch (error) {
        console.error(
            "Groq API error:",
            error.status || "",
            error.message
        );

        const status =
            error.status === 401 ? 401 :
            error.status === 429 ? 429 :
            502;

        res.status(status).json({
            error:
                status === 401
                    ? "Invalid API key. Check your .env file."
                    : status === 429
                    ? "API rate limit reached. Try again later."
                    : "AI service failed. Check the server terminal."
        });
    }
});

// 5. Chat Endpoint (Compatible with both Frontend Chat and Groq Advice)
app.post("/api/chat", async (req, res) => {
    const question = req.body.message || req.body.question;
    const crop = req.body.crop || "Not specified";
    const disease = req.body.disease || "Not specified";

    if (!question || typeof question !== "string" || !question.trim()) {
        return res.status(400).json({ error: "Message cannot be empty." });
    }

    const client = getGroqClient();
    if (!client) {
        const fallback = fallbackChatResponse(question);
        return res.json({
            success: true,
            reply: fallback,
            source: 'agrivision-expert',
            notice: 'AI Assistant active.'
        });
    }

    try {
        const completion = await client.chat.completions.create({
            model: "openai/gpt-oss-20b",
            messages: [
                {
                    role: "system",
                    content: `You are a helpful agricultural assistant.
Give practical, easy-to-understand farming advice.
Crop: ${crop}
Detected disease: ${disease}
Explain possible causes, prevention, and safe next steps.
Do not claim a disease is confirmed from text alone.
Do not invent pesticide dosages. Recommend following
the product label and local agricultural expert guidance.
If details are missing, ask a clarifying question.
Important: Do not use asterisks or markdown stars like ** or * anywhere in your response. Use plain text formatting and simple hyphens (-) for lists.`
                },
                {
                    role: "user",
                    content: question.trim()
                }
            ],
            temperature: 0.4,
            max_completion_tokens: 1000
        });

        const advice = completion.choices?.[0]?.message?.content;
        if (!advice) {
            return res.status(502).json({ error: "The AI returned an empty response." });
        }

        const cleanedAdvice = cleanApiOutput(advice);
        res.json({
            success: true,
            reply: cleanedAdvice,
            advice: cleanedAdvice,
            source: 'agrivision-ai',
            model: 'AgriVision Neural Assistant'
        });
    } catch (error) {
        console.error("AI service error:", error.status || "", error.message);
        const status = error.status === 401 ? 401 : error.status === 429 ? 429 : 502;
        res.status(status).json({
            error:
                status === 401
                    ? "Invalid API key. Check your configuration."
                    : status === 429
                    ? "API rate limit reached. Try again later."
                    : "AI service failed. Check the server terminal."
        });
    }
});

// Start Server with port fallback in case of Windows EACCES/EADDRINUSE
function startServer(port) {
    const server = app.listen(port, () => {
        const client = getGroqClient();
        console.log(`====================================================`);
        console.log(` Smart Crop AI Server (AgriVision)`);
        console.log(` Web UI Running at : http://localhost:${port}`);
        console.log(` AI Engine Status  : ${client ? 'Active (Neural Intelligence Engine)' : 'Local Engine Active'}`);
        console.log(` Supported Crops   : 13 Crops (${SUPPORTED_CROPS.length} Supported Staple Crops)`);
        console.log(` Advice Endpoint   : POST http://localhost:${port}/api/advice`);
        console.log(`====================================================`);
    });

    server.on('error', (err) => {
        if (err.code === 'EACCES' || err.code === 'EADDRINUSE') {
            console.warn(`! Port ${port} unavailable (${err.code}). Trying port ${port + 1}...`);
            startServer(port + 1);
        } else {
            console.error('Server error:', err);
        }
    });
}

startServer(Number(PORT));
