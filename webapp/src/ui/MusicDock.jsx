// Music dock (bottom-left, z 11): drop in any audio file, the game measures
// its tempo in-browser (spectral flux + autocorrelation, the same algorithm
// as duck-mod/scripts/beat_track.py), plays it through WebAudio and hands
// the duck to the beat-dance policy whenever it's idle. WASD still steals
// the duck back to the walker mid-song.
import { useRef } from "react";
import Box from "@mui/material/Box";
import { useGame, gameApi } from "../store.js";
import { MONO, ORANGE } from "../theme.js";

const GLASS = "rgba(8, 8, 12, 0.72)";

export default function MusicDock() {
  const dance = useGame((s) => s.dance);
  const bootDone = useGame((s) => s.bootDone);
  const entered = useGame((s) => s.entered);
  const fileRef = useRef(null);

  if (!bootDone || !entered) return null;

  const busy = dance.state === "analyzing";
  const playing = dance.state === "playing";

  const pick = (e) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // same file can be picked again
    if (file) gameApi.startDance?.(file);
  };

  return (
    <Box
      sx={{
        position: "fixed",
        left: 12,
        bottom: 12,
        zIndex: 11,
        fontFamily: MONO,
        fontSize: 12,
        color: "rgba(255,255,255,0.85)",
        background: GLASS,
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 2,
        px: 1.5,
        py: 1,
        maxWidth: 320,
        backdropFilter: "blur(4px)",
      }}
    >
      <input
        ref={fileRef}
        type="file"
        accept="audio/*"
        hidden
        onChange={pick}
      />
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Box
          component="button"
          onClick={() => (playing ? gameApi.stopDance?.() : fileRef.current?.click())}
          disabled={busy}
          sx={{
            fontFamily: MONO,
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: "0.06em",
            color: playing ? "#08080c" : ORANGE,
            background: playing ? ORANGE : "rgba(255,122,47,0.12)",
            border: `1px solid ${ORANGE}`,
            borderRadius: 1.5,
            px: 1.2,
            py: 0.5,
            cursor: busy ? "wait" : "pointer",
            "&:hover": { filter: "brightness(1.15)" },
          }}
        >
          {busy ? "MEASURING…" : playing ? "■ STOP" : "♪ PLAY A SONG"}
        </Box>
        {playing && (
          <Box sx={{ color: ORANGE, whiteSpace: "nowrap" }}>
            {dance.driveBpm?.toFixed(0)} BPM
          </Box>
        )}
      </Box>
      {(playing || dance.state === "error") && (
        <Box
          sx={{
            mt: 0.5,
            color: dance.state === "error" ? "#ff5555" : "rgba(255,255,255,0.55)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={dance.title}
        >
          {dance.state === "error" ? "couldn't play that file" : dance.title}
        </Box>
      )}
      {dance.note && (
        <Box sx={{ mt: 0.5, color: "rgba(255,200,120,0.8)" }}>{dance.note}</Box>
      )}
      {playing && (
        <Box sx={{ mt: 0.5, color: "rgba(255,255,255,0.4)" }}>
          WASD walks · hands off to dance
        </Box>
      )}
    </Box>
  );
}
