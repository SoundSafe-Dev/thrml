"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

type AlgorithmPoint = {
  x: number;
  y: number;
  z: number;
  label: string;
  metrics: Record<string, number | string>;
};

type AlgorithmDatum = {
  id: string;
  name: string;
  summary: string;
  kpis: Record<string, number | string>;
  points: AlgorithmPoint[];
};

type DemoPayload = {
  generated_at: string;
  bench_config: Record<string, number>;
  algorithms: AlgorithmDatum[];
};

type PointSelection = {
  algorithmId: string;
  label: string;
  metrics: Record<string, number | string>;
};

const colorForIndex = (index: number, total: number): THREE.Color => {
  const color = new THREE.Color();
  const hue = (index / Math.max(total, 1)) % 1;
  color.setHSL(hue, 0.55, 0.6);
  return color;
};

function useDemoData(): DemoPayload | null {
  const [data, setData] = useState<DemoPayload | null>(null);

  useEffect(() => {
    let isMounted = true;
    fetch("/data/algorithm_metrics.json")
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load demo data: ${response.status}`);
        }
        return response.json();
      })
      .then((payload: DemoPayload) => {
        if (isMounted) {
          setData(payload);
        }
      })
      .catch((error) => {
        console.error(error);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return data;
}

const InfoPanel = ({
  datum,
  isActive,
  onFocus,
}: {
  datum: AlgorithmDatum;
  isActive: boolean;
  onFocus: (id: string) => void;
}) => {
  const sortedKpis = useMemo(
    () =>
      Object.entries(datum.kpis).sort(([a], [b]) =>
        a.localeCompare(b)
      ),
    [datum.kpis]
  );

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onFocus(datum.id)}
      onFocus={() => onFocus(datum.id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onFocus(datum.id);
        }
      }}
      style={{
        background: "rgba(15, 23, 42, 0.7)",
        borderRadius: "16px",
        padding: "1.25rem",
        border: isActive
          ? "1px solid rgba(96, 165, 250, 0.8)"
          : "1px solid rgba(148, 163, 184, 0.25)",
        boxShadow: isActive
          ? "0 0 0 1px rgba(59, 130, 246, 0.45)"
          : "none",
        backdropFilter: "blur(10px)",
        width: "100%",
        cursor: "pointer",
        transition: "border 0.2s ease, box-shadow 0.2s ease",
        outline: "none",
      }}
    >
      <h2 style={{ margin: "0 0 0.5rem", fontSize: "1.25rem" }}>{datum.name}</h2>
      <p style={{ margin: "0 0 1rem", color: "#cbd5f5" }}>{datum.summary}</p>
      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "0.75rem",
          margin: 0,
        }}
      >
        {sortedKpis.map(([key, value]) => (
          <div key={`${datum.id}-${key}`}>
            <dt style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "#94a3b8" }}>
              {key.replace(/_/g, " ")}
            </dt>
            <dd style={{ margin: 0, fontSize: "1.1rem", fontWeight: 600 }}>
              {typeof value === "number"
                ? value.toLocaleString(undefined, {
                    maximumFractionDigits: 4,
                  })
                : value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
};

const SelectionDetails = ({
  selection,
  algorithm,
}: {
  selection: PointSelection | null;
  algorithm: AlgorithmDatum | undefined;
}) => {
  if (!selection || !algorithm) {
    return (
      <div
        style={{
          background: "rgba(15, 23, 42, 0.65)",
          borderRadius: "16px",
          padding: "1.25rem",
          border: "1px dashed rgba(148, 163, 184, 0.35)",
          textAlign: "center",
          color: "#94a3b8",
        }}
      >
        Click or tap any point in the scene to inspect detailed metrics.
      </div>
    );
  }

  const metricEntries = Object.entries(selection.metrics);

  return (
    <div
      style={{
        background: "rgba(15, 23, 42, 0.8)",
        borderRadius: "16px",
        padding: "1.5rem",
        border: "1px solid rgba(96, 165, 250, 0.45)",
        backdropFilter: "blur(12px)",
      }}
    >
      <h3 style={{ margin: "0 0 0.75rem", fontSize: "1.2rem" }}>Selected node</h3>
      <p style={{ margin: "0 0 0.35rem", color: "#cbd5f5" }}>
        Algorithm: <strong>{algorithm.name}</strong>
      </p>
      <p style={{ margin: "0 0 0.75rem", color: "#cbd5f5" }}>
        Label: <strong>{selection.label}</strong>
      </p>
      <dl
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "0.85rem",
          margin: 0,
        }}
      >
        {metricEntries.map(([key, value]) => (
          <div key={`selected-${key}`}>
            <dt style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "#94a3b8" }}>
              {key.replace(/_/g, " ")}
            </dt>
            <dd style={{ margin: 0, fontSize: "1.1rem", fontWeight: 600 }}>
              {typeof value === "number"
                ? value.toLocaleString(undefined, {
                    maximumFractionDigits: 4,
                  })
                : value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
};

const ThermodynamicCanvas = ({
  data,
  onSelectPoint,
  activeAlgorithmId,
}: {
  data: DemoPayload;
  onSelectPoint: (selection: PointSelection | null) => void;
  activeAlgorithmId: string | null;
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const haloRefs = useRef<Record<string, THREE.Mesh>>(Object.create(null));
  const pointMeshesRef = useRef<THREE.Mesh[]>([]);

  useEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    const container = containerRef.current;
    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0f172a");

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(6, 5, 10);
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.cursor = "grab";
    container.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.9);
    directionalLight.position.set(5, 6, 4);
    scene.add(directionalLight);

    const axes = new THREE.AxesHelper(4);
    scene.add(axes);

    const grid = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    scene.add(grid);

    pointMeshesRef.current = [];
    haloRefs.current = Object.create(null);

    data.algorithms.forEach((algo, index) => {
      const group = new THREE.Group();
      group.userData = { algorithmId: algo.id };
      const baseColor = colorForIndex(index, data.algorithms.length);
      const material = new THREE.MeshStandardMaterial({
        color: baseColor,
        metalness: 0.2,
        roughness: 0.35,
        emissive: new THREE.Color("#000000"),
        emissiveIntensity: 0,
      });
      const geometry = new THREE.SphereGeometry(0.12, 32, 32);

      algo.points.forEach((point) => {
        const mesh = new THREE.Mesh(geometry, material.clone());
        mesh.position.set(point.x, point.y, point.z);
        mesh.userData = { ...point, algorithmId: algo.id };
        pointMeshesRef.current.push(mesh);
        group.add(mesh);
      });

      const haloGeometry = new THREE.RingGeometry(0.4, 0.45, 64);
      const haloMaterial = new THREE.MeshBasicMaterial({
        color: baseColor.getStyle(),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.3,
      });
      const halo = new THREE.Mesh(haloGeometry, haloMaterial);
      halo.rotation.x = Math.PI / 2;
      halo.position.set(index * 1.25 - 1.25, -1.5, 0);
      group.add(halo);
      haloRefs.current[algo.id] = halo;

      group.position.x = index * 1.5 - (data.algorithms.length - 1) * 0.75;
      scene.add(group);
    });

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredMesh: THREE.Mesh | null = null;

    const setHoveredMesh = (mesh: THREE.Mesh | null) => {
      if (hoveredMesh === mesh) {
        return;
      }
      if (hoveredMesh) {
        hoveredMesh.scale.set(1, 1, 1);
        const previousMaterial =
          hoveredMesh.material as THREE.MeshStandardMaterial;
        previousMaterial.emissive.set("#000000");
        previousMaterial.emissiveIntensity = 0;
      }
      hoveredMesh = mesh;
      if (hoveredMesh) {
        hoveredMesh.scale.set(1.4, 1.4, 1.4);
        const nextMaterial = hoveredMesh.material as THREE.MeshStandardMaterial;
        nextMaterial.emissive.set("#60a5fa");
        nextMaterial.emissiveIntensity = 0.6;
      }
      renderer.domElement.style.cursor = hoveredMesh ? "pointer" : "grab";
    };

    const handlePointerMove = (event: PointerEvent) => {
      const bounds = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      mouse.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(pointMeshesRef.current, false);
      if (intersects.length > 0) {
        setHoveredMesh(intersects[0].object as THREE.Mesh);
      } else {
        setHoveredMesh(null);
      }
    };

    const handlePointerLeave = () => {
      setHoveredMesh(null);
    };

    const handleClick = () => {
      if (hoveredMesh) {
        onSelectPoint(hoveredMesh.userData as PointSelection);
      }
    };

    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerleave", handlePointerLeave);
    renderer.domElement.addEventListener("click", handleClick);
    container.addEventListener("mouseleave", handlePointerLeave);

    let frameId: number;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      scene.rotation.y += 0.0025;
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) {
        return;
      }
      const newWidth = containerRef.current.clientWidth || 1;
      const newHeight = containerRef.current.clientHeight || 1;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
      renderer.domElement.removeEventListener("click", handleClick);
      container.removeEventListener("mouseleave", handlePointerLeave);
      renderer.dispose();
      scene.clear();
      container.removeChild(renderer.domElement);
    };
  }, [data, onSelectPoint]);

  useEffect(() => {
    Object.entries(haloRefs.current).forEach(([algorithmId, halo]) => {
      const material = halo.material as THREE.MeshBasicMaterial;
      const isActive = algorithmId === activeAlgorithmId;
      material.opacity = isActive ? 0.6 : 0.25;
      halo.scale.setScalar(isActive ? 1.25 : 1);
    });
  }, [activeAlgorithmId]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "520px",
        borderRadius: "20px",
        overflow: "hidden",
        position: "relative",
        border: "1px solid rgba(148, 163, 184, 0.25)",
        background: "radial-gradient(circle at center, rgba(59, 130, 246, 0.25), rgba(15, 23, 42, 0.85))",
      }}
    />
  );
};

export default function Home() {
  const data = useDemoData();
  const [activeAlgorithmId, setActiveAlgorithmId] = useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<PointSelection | null>(null);

  useEffect(() => {
    if (data && data.algorithms.length > 0 && !activeAlgorithmId) {
      setActiveAlgorithmId(data.algorithms[0].id);
    }
  }, [data, activeAlgorithmId]);

  const handleSelectPoint = useCallback((selection: PointSelection | null) => {
    setSelectedPoint(selection);
    if (selection) {
      setActiveAlgorithmId(selection.algorithmId);
    }
  }, []);

  const handlePanelFocus = useCallback((id: string) => {
    setActiveAlgorithmId(id);
    setSelectedPoint((current) =>
      current && current.algorithmId === id ? current : null
    );
  }, []);

  const activeAlgorithm = useMemo(
    () => data?.algorithms.find((algo) => algo.id === activeAlgorithmId),
    [data, activeAlgorithmId]
  );

  return (
    <main
      style={{
        padding: "3rem 1.5rem 4rem",
        maxWidth: "1200px",
        margin: "0 auto",
        display: "flex",
        flexDirection: "column",
        gap: "2.5rem",
      }}
    >
      <header style={{ textAlign: "center" }}>
        <p
          style={{
            fontSize: "0.85rem",
            textTransform: "uppercase",
            letterSpacing: "0.2em",
            color: "#94a3b8",
            marginBottom: "0.75rem",
          }}
        >
          Thermodynamic Computing Playground
        </p>
        <h1 style={{ margin: 0, fontSize: "2.75rem" }}>
          Next.js + Three.js Visual Proof of ThrML Algorithms
        </h1>
        <p style={{ color: "#cbd5f5", maxWidth: "740px", margin: "1rem auto 0" }}>
          Interact with stochastic resonance, active perception, and Boltzmann
          policy planners in a single view. Each cluster encodes KPIs pulled
          directly from thermal-noise-aware simulations.
        </p>
      </header>

      {data ? (
        <>
          <ThermodynamicCanvas
            data={data}
            onSelectPoint={handleSelectPoint}
            activeAlgorithmId={activeAlgorithmId}
          />
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
              gap: "1.5rem",
            }}
          >
            {data.algorithms.map((algo) => (
              <InfoPanel
                key={algo.id}
                datum={algo}
                isActive={algo.id === activeAlgorithmId}
                onFocus={handlePanelFocus}
              />
            ))}
          </section>
          <SelectionDetails selection={selectedPoint} algorithm={activeAlgorithm} />
          <footer
            style={{
              textAlign: "center",
              color: "#94a3b8",
              fontSize: "0.85rem",
              lineHeight: 1.8,
            }}
          >
            Metrics generated on {new Date(data.generated_at).toLocaleString()} using
            warmup={data.bench_config.warmup}, samples={data.bench_config.samples},
            steps/sample={data.bench_config.steps_per_sample}.
          </footer>
        </>
      ) : (
        <div style={{ textAlign: "center", color: "#94a3b8" }}>
          Loading thermodynamic simulation results...
        </div>
      )}
    </main>
  );
}
