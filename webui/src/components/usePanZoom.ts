import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Колёсико мыши — зум к курсору, перетаскивание фона — панорама.
 * Работает над любым <svg viewBox="0 0 W H">. Клики по узлам не ломаются:
 * панорама стартует только с фона (target === сам svg).
 */
export function usePanZoom(baseW: number, baseH: number) {
  const [vb, setVb] = useState({ x: 0, y: 0, w: baseW, h: baseH });
  const svgRef = useRef<SVGSVGElement | null>(null);
  const vbRef = useRef(vb);
  vbRef.current = vb;
  const baseRef = useRef({ w: baseW, h: baseH });
  baseRef.current = { w: baseW, h: baseH };
  const drag = useRef<{ mx: number; my: number; vx: number; vy: number } | null>(null);

  // нативный wheel-слушатель (passive:false, иначе preventDefault не сработает)
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const px = (e.clientX - rect.left) / rect.width;
      const py = (e.clientY - rect.top) / rect.height;
      const v = vbRef.current;
      const b = baseRef.current;
      const factor = e.deltaY < 0 ? 0.86 : 1 / 0.86; // вверх — приблизить
      let nw = v.w * factor;
      let nh = v.h * factor;
      const minW = b.w * 0.25; // максимум приближения
      const maxW = b.w * 1.0;  // не отдалять дальше исходного кадра
      if (nw < minW) { const s = minW / v.w; nw = v.w * s; nh = v.h * s; }
      if (nw > maxW) { const s = maxW / v.w; nw = v.w * s; nh = v.h * s; }
      const cx = v.x + px * v.w;
      const cy = v.y + py * v.h;
      setVb({ x: cx - px * nw, y: cy - py * nh, w: nw, h: nh });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.target !== e.currentTarget) return; // пан только с фона — не мешаем кликам
    const v = vbRef.current;
    drag.current = { mx: e.clientX, my: e.clientY, vx: v.x, vy: v.y };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!drag.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const v = vbRef.current;
    const dx = ((e.clientX - drag.current.mx) / rect.width) * v.w;
    const dy = ((e.clientY - drag.current.my) / rect.height) * v.h;
    setVb({ ...v, x: drag.current.vx - dx, y: drag.current.vy - dy });
  }, []);

  const onPointerUp = useCallback(() => { drag.current = null; }, []);
  const reset = useCallback(
    () => setVb({ x: 0, y: 0, w: baseRef.current.w, h: baseRef.current.h }), []);

  const b = baseRef.current;
  const zoomed = Math.abs(vb.w - b.w) > 1 || Math.abs(vb.x) > 1 || Math.abs(vb.y) > 1;
  const viewBox = `${vb.x.toFixed(1)} ${vb.y.toFixed(1)} ${vb.w.toFixed(1)} ${vb.h.toFixed(1)}`;
  return { svgRef, viewBox, onPointerDown, onPointerMove, onPointerUp, reset, zoomed };
}
