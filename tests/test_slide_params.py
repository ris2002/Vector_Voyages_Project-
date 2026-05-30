from rag_backend.agent import (
    _extract_slide_count,
    _extract_slides_format,
    _extract_detail_level,
    _extract_points_per_slide,
    _extract_point_length,
)

passed = 0
failed = 0

def check(label, result, expected):
    global passed, failed
    if result == expected:
        passed += 1
        print(f"PASS: {label} -- got '{result}'")
    else:
        failed += 1
        print(f"FAIL: {label} -- expected '{expected}', got '{result}'")

# Slide count
check("7 slides",             _extract_slide_count("make me 7 slides"),         7)
check("10 slide deck",        _extract_slide_count("create a 10 slide deck"),   10)
check("five slides",          _extract_slide_count("give me five slides"),       5)
check("no count mentioned",   _extract_slide_count("make me some slides"),       None)

# Format
check("paragraphs",           _extract_slides_format("use paragraphs"),          "paragraphs")
check("bullet points",        _extract_slides_format("bullet points please"),    "bullets")
check("both format",          _extract_slides_format("use both bullets and paragraphs"), "both")
check("default format",       _extract_slides_format("make me slides"),          "bullets")

# Detail level
check("detailed",             _extract_detail_level("detailed slides"),          "detailed")
check("brief",                _extract_detail_level("brief overview slides"),    "brief")
check("default detail",       _extract_detail_level("make slides"),              "normal")

# Points per slide
check("5 bullets per slide",  _extract_points_per_slide("5 bullets per slide"),  5)
check("3 points each",        _extract_points_per_slide("3 points each slide"),  3)
check("no count",             _extract_points_per_slide("make me slides"),       None)

# Point length
check("short points",         _extract_point_length("short points"),             "short")
check("long detailed",        _extract_point_length("long detailed bullets"),    "long")
check("concise",              _extract_point_length("concise bullets"),          "short")
check("no length hint",       _extract_point_length("make me slides"),           None)

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")