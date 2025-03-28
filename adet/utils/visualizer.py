import numpy as np
import pickle
from detectron2.utils.visualizer import Visualizer
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm

class TextVisualizer(Visualizer):
    def __init__(self, image, metadata, instance_mode, cfg):
        Visualizer.__init__(self, image, metadata, instance_mode=instance_mode)
        self.voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.BATEXT.CUSTOM_DICT
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        if not self.use_customer_dictionary:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        assert(int(self.voc_size - 1) == len(self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1), len(self.CTLABELS))

    def draw_instance_predictions(self, predictions):
        if self.use_polygon:
            ctrl_pnts = predictions.polygons.numpy()
        else:
            ctrl_pnts = predictions.beziers.numpy()
        scores = predictions.scores.tolist()
        recs = predictions.recs

        self.overlay_instances(ctrl_pnts, recs, scores)

        return self.output

    def _ctrl_pnt_to_poly(self, pnt):
        if self.use_polygon:
            points = pnt.reshape(-1, 2)
        else:
            # bezier to polygon
            u = np.linspace(0, 1, 20)
            pnt = pnt.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
            points = np.outer((1 - u) ** 3, pnt[:, 0]) \
                + np.outer(3 * u * ((1 - u) ** 2), pnt[:, 1]) \
                + np.outer(3 * (u ** 2) * (1 - u), pnt[:, 2]) \
                + np.outer(u ** 3, pnt[:, 3])
            points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

        return points

    def _decode_recognition(self, rec):
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if self.voc_size == 96:
                    s += self.CTLABELS[c]
                else:
                    s += str(chr(self.CTLABELS[c]))
            elif c == self.voc_size -1:
                s += u'口'
        return s

#     def _decode_recognition(self, rec):
#         CTLABELS = [" ",    "!",    '"',     "#",     "$",    "%",    "&",    "'",    "(",    ")",    "*",    "+",    ",",
#     "-",
#     ".",
#     "/",
#     "0",
#     "1",
#     "2",
#     "3",
#     "4",
#     "5",
#     "6",
#     "7",
#     "8",
#     "9",
#     ":",
#     ";",
#     "<",
#     "=",
#     ">",
#     "?",
#     "@",
#     "A",
#     "B",
#     "C",
#     "D",
#     "E",
#     "F",
#     "G",
#     "H",
#     "I",
#     "J",
#     "K",
#     "L",
#     "M",
#     "N",
#     "O",
#     "P",
#     "Q",
#     "R",
#     "S",
#     "T",
#     "U",
#     "V",
#     "W",
#     "X",
#     "Y",
#     "Z",
#     "[",
#     "\\",
#     "]",
#     "^",
#     "_",
#     "`",
#     "a",
#     "b",
#     "c",
#     "d",
#     "e",
#     "f",
#     "g",
#     "h",
#     "i",
#     "j",
#     "k",
#     "l",
#     "m",
#     "n",
#     "o",
#     "p",
#     "q",
#     "r",
#     "s",
#     "t",
#     "u",
#     "v",
#     "w",
#     "x",
#     "y",
#     "z",
#     "{",
#     "|",
#     "}",
#     "~",
#     "ˋ",
#     "ˊ",
#     "﹒",
#     "ˀ",
#     "˜",
#     "ˇ",
#     "ˆ",
#     "˒",
#     "‑",
# ]       
#         s = ''
#         for c in rec:
#             c = int(c)
#             if 0<c < len(CTLABELS):
#                 # if not "ctw1500" in self.dataset_name and not "vintext" in self.dataset_name:
#                 #     if CTLABELS[c-1] in "_0123456789abcdefghijklmnopqrstuvwxyz":
#             #     s += CTLABELS[c]
#             # else:
#                 s += CTLABELS[c]
#             elif c == 104:
#                 s += "口"
#         #if "vintext" in self.dataset_name:
#         s = _vintext_decoder(s)
#         return s

    

    def _ctc_decode_recognition(self, rec):
        # ctc decoding
        last_char = False
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            elif c == self.voc_size -1:
                s += u'口'
            else:
                last_char = False
        return s

    

    def overlay_instances(self, ctrl_pnts, recs, scores, alpha=0.5):
        color = (0, 0.9, 0)

        for ctrl_pnt, rec, score in zip(ctrl_pnts, recs, scores):
            polygon = self._ctrl_pnt_to_poly(ctrl_pnt)
            self.draw_polygon(polygon, color, alpha=0.1)

            # draw text in the top left corner
            text = self._decode_recognition(rec)
            # print(text)
            # if text == "Lets":
            #     text = "Let's"


            text = "{:.3f}: {}".format(score, text)
            text = "{}".format(text)

            
            lighter_color = self._change_color_brightness(color, brightness_factor=0.1)
            text_pos = polygon[15]
            horiz_align = "right"
            font_size = self._default_font_size

            self.draw_text(
                text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=13,
                draw_chinese=False if self.voc_size == 96 else True
            )
    

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
        draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        
        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output

dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"


def make_groups():
    #dictionary = "aàáạảãâầấậẩẫăằắặẳẵAÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪeèéẹẻẽêềếệểễEÈÉẸẺẼÊỀẾỆỂỄoòóọỏõôồốộổỗơờớợởỡOÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠiìíịỉĩIÌÍỊỈĨuùúụủũưừứựửữƯỪỨỰỬỮUÙÚỤỦŨyỳýỵỷỹYỲÝỴỶỸ"

    groups = []
    i = 0
    while i < len(dictionary) - 5:
        group = [c for c in dictionary[i : i + 6]]
        i += 6
        groups.append(group)
    return groups


groups = make_groups()

TONES = ["", "ˋ", "ˊ", "﹒", "ˀ", "˜"]
SOURCES = ["ă", "â", "Ă", "Â", "ê", "Ê", "ô", "ơ", "Ô", "Ơ", "ư", "Ư", "Đ", "đ"]
TARGETS = ["aˇ", "aˆ", "Aˇ", "Aˆ", "eˆ", "Eˆ", "oˆ", "o˒", "Oˆ", "O˒", "u˒", "U˒", "D-", "d‑"]


def correct_tone_position(word):
    word = word[:-1]
    if len(word) < 2:
        pass
    first_ord_char = ""
    second_order_char = ""
    for char in word:
        for group in groups:
            if char in group:
                second_order_char = first_ord_char
                first_ord_char = group[0]
    if word[-1] == first_ord_char and second_order_char != "":
        pair_chars = ["qu", "Qu", "qU", "QU", "gi", "Gi", "gI", "GI"]
        for pair in pair_chars:
            if pair in word and second_order_char in ["u", "U", "i", "I"]:
                return first_ord_char
        return second_order_char
    return first_ord_char


def _vintext_decoder(recognition):
    for char in TARGETS:
        recognition = recognition.replace(char, SOURCES[TARGETS.index(char)])
    if len(recognition) < 1:
        return recognition
    if recognition[-1] in TONES:
        if len(recognition) < 2:
            return recognition
        replace_char = correct_tone_position(recognition)
        tone = recognition[-1]
        recognition = recognition[:-1]
        for group in groups:
            if replace_char in group:
                recognition = recognition.replace(replace_char, group[TONES.index(tone)])
    return recognition  