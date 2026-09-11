"""
Swap the face in the existing poster for the one in the personal photo.

Everything else in the poster — the classroom, the hoodie and lanyard, the
laptop, the banners, the axis medallions — is left byte-for-byte untouched.
Only the face region is replaced.

Method: detect the eyes in both images, solve the similarity transform that
carries one eye pair onto the other, warp the photo into poster space, match
its LAB statistics to the poster's lighting, and composite through a feathered
ellipse clipped to both figures' mattes so nothing bleeds past the hairline.

    pip install opencv-python-headless "rembg[cpu]" pillow numpy
    python swap_face.py
"""
import cv2
import numpy as np

SRC_POSTER = "source-poster.png"
SRC_PHOTO  = "source-photo.jpg"
OUT        = "poster-with-personal-photo.png"

# Eye centres (x, y), from cv2 Haar cascades — re-run the detector if you
# swap in a different photo.
PHOTO_EYES  = ((342.0, 305.0), (477.0, 282.0))
POSTER_EYES = ((597.0, 310.0), (686.0, 290.0))

# Face ellipse in poster coordinates: centre, semi-axes, feather sigma.
# Keep the top below the poster's hairline so its own hair and ears survive.
ELLIPSE = (655, 352, 112, 132)
FEATHER = 12


def similarity(src_pair, dst_pair):
    (sl, sr), (dl, dr) = src_pair, dst_pair
    sv = (sr[0] - sl[0], sr[1] - sl[1])
    dv = (dr[0] - dl[0], dr[1] - dl[1])
    scale = np.hypot(*dv) / np.hypot(*sv)
    rot = np.degrees(np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0]))
    smid = ((sl[0] + sr[0]) / 2, (sl[1] + sr[1]) / 2)
    dmid = ((dl[0] + dr[0]) / 2, (dl[1] + dr[1]) / 2)
    M = cv2.getRotationMatrix2D(smid, -rot, scale)
    M[0, 2] += dmid[0] - smid[0]
    M[1, 2] += dmid[1] - smid[1]
    return M


def matte(path, session):
    from rembg import remove
    from PIL import Image
    cut = remove(Image.open(path).convert("RGB"), session=session,
                 post_process_mask=True)
    return np.array(cut.getchannel("A"))


def main():
    from rembg import new_session
    session = new_session("u2net_human_seg")

    poster = cv2.imread(SRC_POSTER)
    photo = cv2.imread(SRC_PHOTO)
    h, w = poster.shape[:2]

    M = similarity(PHOTO_EYES, POSTER_EYES)
    warp = cv2.warpAffine(photo, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)
    photo_a = cv2.warpAffine(matte(SRC_PHOTO, session), M, (w, h),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    poster_a = matte(SRC_POSTER, session)

    cx, cy, a, b = ELLIPSE
    ell = np.zeros((h, w), np.uint8)
    cv2.ellipse(ell, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    inside = cv2.erode((poster_a > 128).astype(np.uint8) * 255,
                       np.ones((7, 7), np.uint8))
    mask = cv2.bitwise_and(ell, inside)

    sel = mask > 0
    src = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(poster, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        ms, ss = src[:, :, c][sel].mean(), src[:, :, c][sel].std()
        md, sd = dst[:, :, c][sel].mean(), dst[:, :, c][sel].std()
        src[:, :, c] = (src[:, :, c] - ms) * (sd / max(ss, 1e-3)) + md
    graded = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), FEATHER)
    for m in (poster_a, photo_a):
        alpha *= cv2.GaussianBlur((m > 128).astype(np.float32), (0, 0), 2.5)
    alpha = np.clip(alpha, 0, 1)[:, :, None]

    cv2.imwrite(OUT, (graded * alpha + poster * (1 - alpha)).astype(np.uint8))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
