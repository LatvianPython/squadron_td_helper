import glob
import pathlib
import time

import cv2
import numpy as np
import pandas as pd
import terminaltables
from PIL import ImageGrab
from blessed import Terminal
from matplotlib import colors

term = Terminal()


def preprocess_image(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def scale_image(image, scaling_factor):
    width = int(image.shape[1] * scaling_factor / 100)
    height = int(image.shape[0] * scaling_factor / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def screen_grab(x, y, w, h):
    return np.array(ImageGrab.grab(bbox=(x, y, x + w, y + h)))


def to_separate_chars(image):
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for num, c in enumerate(reversed(contours)):
        x, y, w, h = cv2.boundingRect(c)
        single_char = 255 - threshold[y : y + h, x : x + w]
        yield single_char


def make_image_detector(templates):
    def _image_detector(image):
        for key, template in templates.items():
            enlarged = cv2.copyMakeBorder(
                image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
            )

            if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
                continue

            res = cv2.matchTemplate(enlarged, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.65
            loc = np.where(res >= threshold)
            xs, ys = loc
            if xs.size and ys.size:
                return key

    return _image_detector


def load_templates(path):
    return {pathlib.Path(file).stem: cv2.imread(file, 0) for file in glob.glob(path)}


number_detector = make_image_detector(load_templates("./digits/*"))
builder_detector = make_image_detector(load_templates("./builders/*"))


def prepare_screen_grab(image, threshold=200):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = preprocess_image(image)

    image[np.where((image <= [threshold]))] = [0]
    image = ~image
    image = scale_image(image, 250)

    return image


def detect_number(image):
    return int(
        "".join([number_detector(img) or "" for img in to_separate_chars(image)]) or 0
    )


# resource_offset = 100
resource_offset = 50
# resource_offset = 0


def minerals():
    return detect_number(
        prepare_screen_grab(screen_grab(2020 - resource_offset, 47, 125, 30))
    )


def vespene_gas():
    return detect_number(
        prepare_screen_grab(screen_grab(2185 - resource_offset, 47, 125, 30))
    )


def builder():
    return builder_detector(prepare_screen_grab(screen_grab(820, 1184, 190, 190), 0))


def format_tier(tower):
    if "u" in tower["tier"]:
        return "  ┕━"
    return f"  {tower['tier']}"


def format_name(tower):
    if pd.isna(tower["good"]):
        return term.normal + tower["name"]
    if tower["good"]:
        return term.green(tower["name"])


def format_special(tower):
    special = ""
    if pd.notna(tower["special"]):
        special += "🐌" * 3
    if tower["dps-bonus-add"] > 0:
        special += "⚔️" * int(tower["dps-bonus-add"] / 100)
    if tower["dps-bonus-mult"] > 1:
        special += "💥" * int(tower["dps-bonus-mult"])
    if tower["attacks"] > 1:
        special += "💣" * 3
    return special


red_green_colormap = colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"]
)


def format_dps(tower):
    color = list(
        int(color * 255) for color in red_green_colormap(tower["scaled_dps_cost"])[:-1]
    )
    return term.color_rgb(*color) + f"{tower['dps']:8.2f}" + term.normal


def format_life(tower):
    color = list(
        int(color * 255) for color in red_green_colormap(tower["scaled_life_cost"])[:-1]
    )
    return term.color_rgb(*color) + f"{tower['life']:4.0f}" + term.normal


def format_range(tower):
    return tower["range"]


def format_cost(tower):
    color = term.limegreen if tower["can_afford"] else term.orangered
    return color(str(tower["cost"]))


def to_console(game_state):
    table_header = [
        "tier",
        term.normal + "name",
        term.normal + "cost",
        term.normal + "dps",
        term.normal + "life",
        "range",
        "special",
    ]

    table = terminaltables.AsciiTable(table_header)

    table.inner_heading_row_border = False
    table.outer_border = False

    table.table_data = [table_header] + [
        [
            format_tier(tower),
            format_name(tower),
            format_cost(tower),
            format_dps(tower),
            format_life(tower),
            format_range(tower),
            format_special(tower),
        ]
        for _, tower in game_state["towers"].sort_values("tier").iterrows()
    ]

    print(term.home + term.clear)
    # print(f"minerals={game_state['minerals']} gas={game_state['vespene_gas']}")
    print(table.table)


def read_towers():
    try:
        return read_towers.towers
    except AttributeError:
        read_towers.towers = pd.read_csv("towers.csv")
        return read_towers()


def current_game_state(last_builder):
    current_builder = builder() or last_builder
    # print(current_builder)

    towers = read_towers()

    current_minerals = minerals()
    current_vespene_gas = vespene_gas()

    available_towers: pd.DataFrame = towers.loc[
        (towers.builder == current_builder)
    ].copy()

    available_towers["can_afford"] = available_towers.cost <= current_minerals

    return (
        current_builder,
        {
            "builder": current_builder,
            "minerals": current_minerals,
            "vespene_gas": current_vespene_gas,
            "towers": available_towers,
        },
    )


def run():
    last_builder = None
    while True:
        last_builder, game_state = current_game_state(last_builder)
        to_console(game_state)
        time.sleep(1)


if __name__ == "__main__":
    with term.fullscreen(), term.hidden_cursor(), term.cbreak():
        run()
