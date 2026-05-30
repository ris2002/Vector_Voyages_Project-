/**
 * pptx_builder.js
 * Usage: node pptx_builder.js <input_json_path> <output_pptx_path>
 *
 * Reads slide data JSON produced by tools.generate_slides() and
 * creates a polished .pptx using PptxGenJS.
 *
 * JSON shape:
 * {
 *   "format":  "bullets" | "paragraphs" | "both",
 *   "detail":  "brief" | "normal" | "detailed",
 *   "slides": [
 *     {
 *       "title": "...",
 *       "content": ["bullet 1", "bullet 2"],   // used for bullets / both
 *       "paragraph": "Full paragraph text.",   // used for paragraphs / both
 *       "speaker_note": "..."
 *     }
 *   ]
 * }
 */

"use strict";

const fs      = require("fs");
const path    = require("path");
const pptxgen = require("pptxgenjs");

// ── CLI args ──────────────────────────────────────────────────────────────────
const [,, inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  console.error("Usage: node pptx_builder.js <input.json> <output.pptx>");
  process.exit(1);
}

const data      = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const FORMAT    = data.format  || "bullets";
const DETAIL    = data.detail  || "normal";
const slides    = data.slides  || [];

// ── Design tokens ─────────────────────────────────────────────────────────────
const C = {
  // Ocean Gradient palette
  primary   : "065A82",   // deep blue  — title/header bg
  secondary : "1C7293",   // teal       — accent strips
  accent    : "21295C",   // midnight   — dark bg slides
  white     : "FFFFFF",
  offWhite  : "F0F6FA",
  bodyText  : "1E293B",
  mutedText : "64748B",
  bullet    : "0891B2",   // bright teal bullet dots
  noteText  : "475569",
};

const FONT_TITLE  = "Georgia";
const FONT_BODY   = "Calibri";
const W           = 10;     // slide width  (inches, LAYOUT_16x9)
const H           = 5.625;  // slide height

// ── Helpers ───────────────────────────────────────────────────────────────────
function makeShadow() {
  return { type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 135 };
}

/**
 * Add a dark-background "hero" slide (title + subtitle bar).
 * Used for slide 1 and slide 10.
 */
function addHeroSlide(pres, slideData, isTitle) {
  const sl = pres.addSlide();
  sl.background = { color: C.accent };

  // Large circle decoration top-right
  sl.addShape(pres.shapes.OVAL, {
    x: 7.8, y: -1.2, w: 4, h: 4,
    fill: { color: C.secondary, transparency: 75 }, line: { color: C.secondary, transparency: 75 },
  });

  // Small accent bar left edge
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.6, w: 0.12, h: 2.4,
    fill: { color: C.bullet }, line: { color: C.bullet },
  });

  // Title
  sl.addText(slideData.title || "Presentation", {
    x: 0.35, y: 1.5, w: 8.5, h: 1.4,
    fontFace: FONT_TITLE, fontSize: isTitle ? 36 : 30, bold: true,
    color: C.white, align: "left", valign: "middle", margin: 0,
  });

  // Content as subtitle line(s) on hero slides
  const body = slideData.paragraph ||
    (slideData.content || []).join("  •  ");
  if (body) {
    sl.addText(body, {
      x: 0.35, y: 3.1, w: 8.5, h: 1.2,
      fontFace: FONT_BODY, fontSize: 14,
      color: C.offWhite, align: "left", valign: "top", margin: 0,
    });
  }

  addSpeakerNote(sl, slideData.speaker_note);
  return sl;
}

/**
 * Add a light-background content slide with bullet points.
 */
function addBulletSlide(pres, slideData) {
  const sl = pres.addSlide();
  sl.background = { color: C.offWhite };

  // Header strip
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 1.0,
    fill: { color: C.primary }, line: { color: C.primary },
  });

  // Title in header
  sl.addText(slideData.title || "", {
    x: 0.4, y: 0, w: W - 0.8, h: 1.0,
    fontFace: FONT_TITLE, fontSize: 22, bold: true,
    color: C.white, align: "left", valign: "middle", margin: 0,
  });

  // Accent bottom strip
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.25, w: W, h: 0.25,
    fill: { color: C.secondary }, line: { color: C.secondary },
  });

  // Bullet points
  const bullets = slideData.content || [];
  if (bullets.length > 0) {
    const bulletItems = bullets.map((txt, i) => ({
      text: txt,
      options: {
        bullet: { color: C.bullet },
        color: C.bodyText,
        fontSize: bulletFontSize(DETAIL),
        breakLine: i < bullets.length - 1,
        paraSpaceAfter: DETAIL === "detailed" ? 4 : 6,
      },
    }));

    sl.addText(bulletItems, {
      x: 0.5, y: 1.2, w: W - 1.0, h: H - 1.7,
      fontFace: FONT_BODY, valign: "top",
    });
  }

  addSpeakerNote(sl, slideData.speaker_note);
  return sl;
}

/**
 * Add a light-background content slide with a paragraph.
 * Supports multiple paragraphs separated by \n\n — each rendered as a distinct block.
 */
function addParagraphSlide(pres, slideData) {
  const sl = pres.addSlide();
  sl.background = { color: C.white };

  // Left accent column
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.3, h: H,
    fill: { color: C.primary }, line: { color: C.primary },
  });

  // Title
  sl.addText(slideData.title || "", {
    x: 0.55, y: 0.25, w: W - 0.85, h: 0.8,
    fontFace: FONT_TITLE, fontSize: 24, bold: true,
    color: C.primary, align: "left", valign: "middle", margin: 0,
  });

  // Thin rule under title
  sl.addShape(pres.shapes.LINE, {
    x: 0.55, y: 1.1, w: W - 1.0, h: 0,
    line: { color: C.secondary, width: 1.5 },
  });

  // Split into individual paragraphs on \n\n
  const paraRaw = slideData.paragraph || "";
  const paraBlocks = paraRaw.split(/\n\n+/).map(p => p.trim()).filter(Boolean);

  if (paraBlocks.length > 0) {
    const contentH = H - 1.7;          // total available height
    const blockH   = contentH / paraBlocks.length;
    const fontSize = paraFontSize(DETAIL);

    paraBlocks.forEach((block, idx) => {
      const yPos = 1.25 + idx * blockH;

      // Subtle separator line between paragraphs (skip before first)
      if (idx > 0) {
        sl.addShape(pres.shapes.LINE, {
          x: 0.55, y: yPos - 0.08, w: W - 1.1, h: 0,
          line: { color: C.mutedText, width: 0.5, transparency: 60 },
        });
      }

      sl.addText(block, {
        x: 0.55, y: yPos, w: W - 1.0, h: blockH - 0.1,
        fontFace: FONT_BODY, fontSize,
        color: C.bodyText, align: "left", valign: "top",
        lineSpacingMultiple: 1.35,
      });
    });
  }

  addSpeakerNote(sl, slideData.speaker_note);
  return sl;
}

/**
 * Add a split slide: bullets on left, paragraph on right.
 * Used when format === 'both'.
 */
function addBothSlide(pres, slideData) {
  const sl = pres.addSlide();
  sl.background = { color: C.offWhite };

  // Header
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.9,
    fill: { color: C.primary }, line: { color: C.primary },
  });
  sl.addText(slideData.title || "", {
    x: 0.4, y: 0, w: W - 0.8, h: 0.9,
    fontFace: FONT_TITLE, fontSize: 20, bold: true,
    color: C.white, align: "left", valign: "middle", margin: 0,
  });

  // Divider
  sl.addShape(pres.shapes.LINE, {
    x: 5.05, y: 1.05, w: 0, h: H - 1.4,
    line: { color: C.secondary, width: 1.2 },
  });

  // Left: bullets
  const bullets = slideData.content || [];
  if (bullets.length > 0) {
    const items = bullets.map((txt, i) => ({
      text: txt,
      options: {
        bullet: { color: C.bullet },
        color: C.bodyText,
        fontSize: 13,
        breakLine: i < bullets.length - 1,
        paraSpaceAfter: 5,
      },
    }));
    sl.addText(items, {
      x: 0.35, y: 1.1, w: 4.5, h: H - 1.5,
      fontFace: FONT_BODY, valign: "top",
    });
  }

  // Right: paragraph(s) — split on \n\n for multiple blocks
  const paraRaw = slideData.paragraph || "";
  const paraBlocks = paraRaw.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  if (paraBlocks.length > 0) {
    const contentH = H - 1.5;
    const blockH   = contentH / paraBlocks.length;

    paraBlocks.forEach((block, idx) => {
      const yPos = 1.1 + idx * blockH;

      if (idx > 0) {
        sl.addShape(pres.shapes.LINE, {
          x: 5.25, y: yPos - 0.07, w: 4.3, h: 0,
          line: { color: C.mutedText, width: 0.5, transparency: 60 },
        });
      }

      sl.addText(block, {
        x: 5.25, y: yPos, w: 4.4, h: blockH - 0.1,
        fontFace: FONT_BODY, fontSize: 13,
        color: C.bodyText, align: "left", valign: "top",
        lineSpacingMultiple: 1.3,
      });
    });
  }

  // Bottom strip
  sl.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: H - 0.2, w: W, h: 0.2,
    fill: { color: C.secondary }, line: { color: C.secondary },
  });

  addSpeakerNote(sl, slideData.speaker_note);
  return sl;
}

function bulletFontSize(detail) {
  if (detail === "brief")    return 17;
  if (detail === "detailed") return 13;
  return 15;
}
function paraFontSize(detail) {
  if (detail === "brief")    return 16;
  if (detail === "detailed") return 13;
  return 14;
}

function addSpeakerNote(sl, note) {
  if (note) sl.addNotes(note);
}

// ── Build presentation ────────────────────────────────────────────────────────
async function build() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.title  = slides[0]?.title || "Research Paper Presentation";
  pres.author = "Vector Voyagers";

  slides.forEach((slideData, idx) => {
    const isHero = idx === 0 || idx === slides.length - 1;

    if (isHero) {
      addHeroSlide(pres, slideData, idx === 0);
      return;
    }

    if (FORMAT === "paragraphs") {
      addParagraphSlide(pres, slideData);
    } else if (FORMAT === "both") {
      addBothSlide(pres, slideData);
    } else {
      // bullets (default)
      addBulletSlide(pres, slideData);
    }
  });

  await pres.writeFile({ fileName: outputPath });
  console.log(`OK: ${outputPath}`);
}

build().catch(err => {
  console.error("PPTX build failed:", err.message);
  process.exit(1);
});
