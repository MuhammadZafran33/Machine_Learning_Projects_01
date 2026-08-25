'use client';

import { useState, useRef, useEffect } from 'react';
import { HelpCircle, TrendingUp, Handshake, Megaphone, Send, Loader2 } from 'lucide-react';

const PERSONAS = {
  general: {
    label: 'General Advisor',
    icon: HelpCircle,
    tagline: 'Comprehensive advice on real estate — from legalities to client management — tailored to your needs.',
    system: 'You are a General Real Estate Advisor AI inside a product called ESTI. You give comprehensive, practical guidance on real estate topics: legalities, client management, listings, market trends. Keep answers structured, concrete, and under 150 words unless the user explicitly asks for more depth. No preamble, no "as an AI" disclaimers.',
    suggestions: [
      'How can I increase my property sales in a competitive market?',
      'What should I consider when managing multiple property listings?',
    ],
  },
  sales: {
    label: 'Sales Advisor',
    icon: TrendingUp,
    tagline: 'Boost your property sales with expert tips and proven strategies for real estate professionals.',
    system: 'You are a Real Estate Sales Advisor AI inside a product called ESTI. You help agents and property owners boost sales: listing optimization, lead generation, pricing strategy, staging advice. Keep answers structured, concrete, and under 150 words unless the user explicitly asks for more depth. No preamble, no "as an AI" disclaimers.',
    suggestions: [
      'How do I write listing descriptions that actually convert?',
      "What's the best way to generate more buyer leads this quarter?",
    ],
  },
  negotiation: {
    label: 'Negotiation Expert',
    icon: Handshake,
    tagline: 'Master negotiation with advice on closing deals, overcoming objections, and maximizing value.',
    system: 'You are a Real Estate Negotiation Expert AI inside a product called ESTI. You help agents close deals: handling objections, countering lowball offers, multiple-offer situations, concrete example phrasing. Keep answers structured, concrete, and under 150 words unless the user explicitly asks for more depth. No preamble, no "as an AI" disclaimers.',
    suggestions: [
      'How do I handle a lowball offer professionally?',
      'What are effective closing techniques for a hesitant buyer?',
    ],
  },
  marketing: {
    label: 'Marketing Guru',
    icon: Megaphone,
    tagline: 'Elevate your marketing with creative campaigns, branding insights, and social media strategies.',
    system: 'You are a Real Estate Marketing Guru AI inside a product called ESTI. You help agents with marketing campaigns, personal branding, and social media strategy that attracts clients. Keep answers structured, concrete, and under 150 words unless the user explicitly asks for more depth. No preamble, no "as an AI" disclaimers.',
    suggestions: [
      'What social media content actually works for real estate?',
      'How do I build a personal brand as a new agent?',
    ],
  },
};

const PERSONA_KEYS = Object.keys(PERSONAS);

const BUBBLE_GRADIENTS = [
  'conic-gradient(from 0deg, rgba(232,121,249,0.5), rgba(139,92,246,0.5), rgba(96,165,250,0.45), rgba(236,72,153,0.5), rgba(232,121,249,0.5))',
  'conic-gradient(from 90deg, rgba(167,139,250,0.5), rgba(244,114,182,0.45), rgba(129,140,248,0.5), rgba(196,181,253,0.45), rgba(167,139,250,0.5))',
  'conic-gradient(from 200deg, rgba(99,102,241,0.45), rgba(232,121,249,0.45), rgba(59,130,246,0.45), rgba(217,70,239,0.45), rgba(99,102,241,0.45))',
];

function BackgroundBlobs() {
  const blobRefs = useRef([]);
  const stateRef = useRef([
    { x: 22, y: 25, vx: 0.018, vy: 0.012, r: 18 },
    { x: 72, y: 68, vx: -0.014, vy: 0.016, r: 17 },
    { x: 55, y: 38, vx: 0.012, vy: -0.015, r: 15 },
  ]);
  const rafRef = useRef(null);

  useEffect(() => {
    const tick = () => {
      const blobs = stateRef.current;

      blobs.forEach((b) => {
        b.x += b.vx;
        b.y += b.vy;
        if (b.x < b.r || b.x > 100 - b.r) b.vx *= -1;
        if (b.y < b.r || b.y > 100 - b.r) b.vy *= -1;
        b.x = Math.min(Math.max(b.x, b.r), 100 - b.r);
        b.y = Math.min(Math.max(b.y, b.r), 100 - b.r);
      });

      for (let i = 0; i < blobs.length; i++) {
        for (let j = i + 1; j < blobs.length; j++) {
          const a = blobs[i];
          const b = blobs[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
          const minDist = (a.r + b.r) * 0.65;
          if (dist < minDist) {
            const overlap = (minDist - dist) / minDist;
            const pushX = (dx / dist) * overlap * 0.06;
            const pushY = (dy / dist) * overlap * 0.06;
            a.vx -= pushX;
            a.vy -= pushY;
            b.vx += pushX;
            b.vy += pushY;
          }
        }
      }

      blobs.forEach((b) => {
        b.vx *= 0.995;
        b.vy *= 0.995;
        const speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy);
        const maxSpeed = 0.06;
        if (speed > maxSpeed) {
          b.vx = (b.vx / speed) * maxSpeed;
          b.vy = (b.vy / speed) * maxSpeed;
        }
      });

      blobRefs.current.forEach((el, i) => {
        if (el) {
          el.style.left = `${blobs[i].x}%`;
          el.style.top = `${blobs[i].y}%`;
        }
      });

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0 esti-color-cycle">
      {stateRef.current.map((b, i) => (
        <div
          key={i}
          ref={(el) => (blobRefs.current[i] = el)}
          className="absolute"
          style={{
            left: `${b.x}%`,
            top: `${b.y}%`,
            width: `${b.r * 2}vmax`,
            height: `${b.r * 2}vmax`,
            transform: 'translate(-50%, -50%)',
          }}
        >
          <div
            className="w-full h-full rounded-full blur-2xl esti-blob-spin"
            style={{
              background: BUBBLE_GRADIENTS[i],
              mixBlendMode: 'screen',
              animationDelay: `${i * -5}s`,
            }}
          />
        </div>
      ))}
    </div>
  );
}

function CursorDust() {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const rafRef = useRef(null);
  const lastSpawnRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const colors = ['232,121,249', '167,139,250', '196,181,253', '236,72,153'];

    const handleMove = (e) => {
      const now = performance.now();
      if (now - lastSpawnRef.current < 12) return;
      lastSpawnRef.current = now;

      for (let i = 0; i < 4; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 1.4 + 0.2;
        particlesRef.current.push({
          x: e.clientX + (Math.random() - 0.5) * 10,
          y: e.clientY + (Math.random() - 0.5) * 10,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 0.3,
          size: Math.random() * 1.1 + 0.4,
          alpha: Math.random() * 0.5 + 0.5,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }

      if (particlesRef.current.length > 260) {
        particlesRef.current.splice(0, particlesRef.current.length - 260);
      }
    };
    window.addEventListener('mousemove', handleMove);

    const tick = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particlesRef.current.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.vx *= 0.96;
        p.vy *= 0.96;
        p.alpha -= 0.025;

        if (p.alpha > 0) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
          ctx.fill();
        }
      });
      particlesRef.current = particlesRef.current.filter((p) => p.alpha > 0);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMove);
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 pointer-events-none z-50" />;
}

function Nav({ view, setView, goToAbout }) {
  return (
    <nav className="flex items-center justify-between px-6 sm:px-10 py-5 shrink-0">
      <button onClick={() => setView('landing')} className="flex items-center gap-3 cursor-pointer group">
        <span className="relative w-9 h-9 rounded-full p-[2px] bg-gradient-to-br from-fuchsia-400 via-purple-500 to-violet-700 shadow-lg shadow-purple-950/60 ring-1 ring-white/10 group-hover:ring-white/30 transition-all">
          <img
            src="/profile.jpg"
            alt="Muhammad Zafran"
            className="w-full h-full rounded-full object-cover"
          />
          <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 ring-2 ring-purple-950" />
        </span>
        <span className="font-bold text-lg tracking-tight bg-gradient-to-r from-white to-violet-200 bg-clip-text text-transparent">
          ESTI
        </span>
      </button>
      <div className="hidden sm:flex items-center gap-8 text-sm">
        <button
          onClick={() => setView('landing')}
          className={
            (view === 'landing' ? 'text-violet-300 font-medium' : 'text-violet-200/60 hover:text-violet-200') +
            ' cursor-pointer transition-colors'
          }
        >
          Home
        </button>
        <button onClick={goToAbout} className="text-violet-200/60 hover:text-violet-200 cursor-pointer transition-colors">
          About
        </button>
        <button
          onClick={() => setView('chat')}
          className={
            (view === 'chat' ? 'text-violet-300 font-medium' : 'text-violet-200/60 hover:text-violet-200') +
            ' cursor-pointer transition-colors'
          }
        >
          Services
        </button>
      </div>
      <div className="flex items-center gap-3">
        <button className="cursor-pointer px-4 py-1.5 text-sm rounded-lg border border-violet-500/50 text-violet-100 hover:bg-violet-900/40 hover:border-violet-400 transition-colors">
          Log In
        </button>
        <button className="cursor-pointer px-4 py-1.5 text-sm rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-medium transition-colors">
          Sign Up
        </button>
      </div>
    </nav>
  );
}

function Landing({ startChat, aboutRef }) {
  return (
    <div className="flex-1 flex flex-col items-center px-6 pb-24 pt-8 text-center">
      <h1 className="max-w-3xl text-3xl sm:text-5xl font-bold leading-tight esti-animate-in">
        Transform Your Real Estate Business With{' '}
        <span className="bg-gradient-to-r from-fuchsia-400 to-violet-400 bg-clip-text text-transparent">
          AI Advisor Chatbots
        </span>
      </h1>
      <p className="max-w-xl mt-5 text-violet-200/70 text-sm sm:text-base esti-animate-in" style={{ animationDelay: '80ms' }}>
        Streamline your real estate operations with tailored advice from specialized AI chatbots
        designed for sales, negotiation, marketing, and more.
      </p>
      <button
        onClick={() => startChat('general')}
        className="cursor-pointer mt-8 px-7 py-3 rounded-lg border border-violet-400/60 text-white font-medium tracking-wide hover:bg-violet-900/40 hover:border-violet-300 transition-colors esti-animate-in"
        style={{ animationDelay: '140ms' }}
      >
        GET STARTED
      </button>

      <div className="w-full max-w-5xl mt-16 text-left">
        <h2 className="text-lg font-semibold mb-5">
          Explore <span className="text-violet-300">AI Chatbots</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {PERSONA_KEYS.map((key, i) => {
            const p = PERSONAS[key];
            const Icon = p.icon;
            return (
              <button
                key={key}
                onClick={() => startChat(key)}
                className="esti-animate-in cursor-pointer text-left p-5 rounded-2xl bg-violet-950/40 border border-violet-700/30 transition-all duration-300 hover:border-violet-400/70 hover:bg-violet-900/40 hover:-translate-y-1.5 hover:shadow-xl hover:shadow-purple-950/50"
                style={{ animationDelay: `${200 + i * 90}ms` }}
              >
                <span className="w-10 h-10 rounded-full bg-violet-600/80 flex items-center justify-center mb-4">
                  <Icon className="w-5 h-5 text-white" />
                </span>
                <p className="font-semibold text-white mb-1.5">{p.label}</p>
                <p className="text-xs text-violet-200/60 leading-relaxed">{p.tagline}</p>
              </button>
            );
          })}
        </div>
      </div>

      <div ref={aboutRef} className="w-full max-w-3xl mt-28 pt-16 border-t border-violet-800/40 text-left scroll-mt-8">
        <h2 className="text-lg font-semibold mb-3">
          About <span className="text-violet-300">ESTI</span>
        </h2>
        <p className="text-sm text-violet-200/70 leading-relaxed">
          ESTI puts a specialized AI advisor behind every part of a real estate deal — sales strategy,
          negotiation tactics, and marketing — so agents and property owners get expert-level guidance
          on demand instead of digging through generic advice. Each advisor is purpose-built for one job
          and trained to give short, actionable answers you can use immediately.
        </p>
      </div>
    </div>
  );
}

function Chat({
  activePersona,
  setActivePersona,
  messages,
  loading,
  error,
  input,
  setInput,
  handleKeyDown,
  sendMessage,
  bottomRef,
}) {
  const persona = PERSONAS[activePersona];
  const currentMessages = messages[activePersona];

  return (
    <div className="flex-1 flex flex-col px-6 sm:px-10 pb-6 min-h-0">
      <div className="flex flex-wrap gap-3 mb-6 shrink-0">
        {PERSONA_KEYS.map((key) => {
          const isActive = key === activePersona;
          return (
            <button
              key={key}
              onClick={() => setActivePersona(key)}
              className={
                'cursor-pointer transition-colors ' +
                (isActive
                  ? 'px-4 py-2 rounded-lg bg-white text-violet-950 text-sm font-medium'
                  : 'px-4 py-2 rounded-lg bg-violet-950/40 border border-violet-700/40 text-violet-200 text-sm hover:border-violet-500/60')
              }
            >
              {PERSONAS[key].label}
            </button>
          );
        })}
      </div>

      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {currentMessages.length === 0 && (
          <div className="max-w-lg">
            <p className="text-violet-200/60 text-sm mb-4">{persona.tagline}</p>
            <div className="flex flex-col items-start gap-2">
              {persona.suggestions.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="cursor-pointer text-left text-sm px-4 py-2.5 rounded-xl bg-violet-900/30 border border-violet-700/40 text-violet-100 hover:border-violet-500/60 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {currentMessages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-md px-4 py-2.5 rounded-2xl rounded-br-sm bg-violet-600 text-white text-sm">
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <div className="max-w-lg px-4 py-3 rounded-2xl rounded-bl-sm bg-violet-950/50 border border-violet-700/30 text-violet-50 text-sm leading-relaxed whitespace-pre-wrap">
                {m.content}
              </div>
            </div>
          )
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="px-4 py-3 rounded-2xl rounded-bl-sm bg-violet-950/50 border border-violet-700/30 flex items-center gap-2 text-violet-300 text-sm">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Thinking…
            </div>
          </div>
        )}

        {error && <p className="text-sm text-rose-300">{error}</p>}
        <div ref={bottomRef} />
      </div>

      <div className="mt-4 shrink-0 flex items-center gap-2 rounded-full border border-violet-500/50 bg-violet-950/40 pl-5 pr-2 py-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Chatbot.."
          className="flex-1 bg-transparent outline-none text-sm text-white placeholder-violet-300/40"
        />
        <button
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          className="cursor-pointer w-9 h-9 shrink-0 rounded-full bg-white text-violet-950 flex items-center justify-center disabled:cursor-not-allowed disabled:opacity-40 hover:bg-violet-100 transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export default function ESTIRealEstateAI() {
  const [view, setView] = useState('landing');
  const [activePersona, setActivePersona] = useState('general');
  const [messages, setMessages] = useState({ general: [], sales: [], negotiation: [], marketing: [] });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [aboutPending, setAboutPending] = useState(false);
  const bottomRef = useRef(null);
  const aboutRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activePersona, loading, view]);

  useEffect(() => {
    if (view === 'landing' && aboutPending) {
      aboutRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setAboutPending(false);
    }
  }, [view, aboutPending]);

  const startChat = (personaKey) => {
    setActivePersona(personaKey);
    setView('chat');
    setError(null);
  };

  const goToAbout = () => {
    if (view !== 'landing') {
      setAboutPending(true);
      setView('landing');
    } else {
      aboutRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const sendMessage = async (textOverride) => {
    const text = (textOverride ?? input).trim();
    if (!text || loading) return;

    const userMsg = { role: 'user', content: text };
    const history = [...messages[activePersona], userMsg];
    setMessages((prev) => ({ ...prev, [activePersona]: history }));
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-6',
          max_tokens: 1000,
          system: PERSONAS[activePersona].system,
          messages: history.map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Request failed');
      }

      const reply = (data.content || [])
        .filter((block) => block.type === 'text')
        .map((block) => block.text)
        .join('\n')
        .trim();

      setMessages((prev) => ({
        ...prev,
        [activePersona]: [
          ...prev[activePersona],
          { role: 'assistant', content: reply || "I couldn't put together an answer that time — try rephrasing." },
        ],
      }));
    } catch (e) {
      setError(e.message === 'Request failed' || !e.message ? 'Could not reach the AI advisor. Check your connection and try again.' : e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      className={
        (view === 'chat' ? 'h-screen overflow-hidden' : 'min-h-screen') +
        ' flex flex-col bg-gradient-to-br from-purple-950 via-violet-950 to-black text-white font-sans'
      }
    >
      <style>{`
        @keyframes esti-fade-up {
          from { opacity: 0; transform: translateY(18px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .esti-animate-in {
          opacity: 0;
          animation: esti-fade-up 0.6s ease-out forwards;
        }
        @keyframes esti-blob-float-a {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(40px, -60px) scale(1.15); }
          66% { transform: translate(-30px, 40px) scale(0.9); }
        }
        @keyframes esti-blob-float-b {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(-50px, 30px) scale(0.95); }
          66% { transform: translate(35px, -45px) scale(1.1); }
        }
        @keyframes esti-blob-float-c {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(25px, 50px) scale(1.2); }
        }
        @keyframes esti-hue-cycle {
          0% { filter: hue-rotate(0deg) saturate(1.3); }
          100% { filter: hue-rotate(360deg) saturate(1.3); }
        }
        .esti-color-cycle {
          animation: esti-hue-cycle 14s linear infinite;
        }
      `}</style>
      <BackgroundBlobs />
      <CursorDust />
      <div className="relative z-10 flex flex-col flex-1 min-h-0">
        <Nav view={view} setView={setView} goToAbout={goToAbout} />
        {view === 'landing' ? (
          <Landing startChat={startChat} aboutRef={aboutRef} />
        ) : (
          <Chat
            activePersona={activePersona}
            setActivePersona={setActivePersona}
            messages={messages}
            loading={loading}
            error={error}
            input={input}
            setInput={setInput}
            handleKeyDown={handleKeyDown}
            sendMessage={sendMessage}
            bottomRef={bottomRef}
          />
        )}
      </div>
    </div>
  );
}
