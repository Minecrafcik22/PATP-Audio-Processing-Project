#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, iirpeak

# --- FIX NA OSTROŚĆ EKRANU (HIGH DPI) ---
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

# Sprawdzamy czy biblioteka yt-dlp jest dostępna
try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False


def convert_to_wav_if_needed(input_path):
    """Konwertuje pliki MP3/M4A/FLAC do tymczasowego WAV za pomocą ffmpeg"""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path, False

    temp_wav = os.path.join(tempfile.gettempdir(), "patp_temp_input.wav")
    
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    ffmpeg_bin = os.path.join(base_dir, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    cmd = [ffmpeg_bin, "-y", "-i", input_path, "-ac", "2", "-ar", "44100", temp_wav]
    
    try:
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.StartupInfo()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, timeout=30)
        return temp_wav, True
    except subprocess.TimeoutExpired:
        raise RuntimeError("Proces konwersji FFmpeg przekroczył limit czasu (30 sekund).")
    except Exception as e:
        raise RuntimeError(f"Błąd konwersji {ext} na WAV. Upewnij się, że plik ffmpeg.exe znajduje się w folderze programu!\n\nSzczegóły: {e}")


def apply_sub_harmonic_synth(signal, sample_rate, target_freq=40.0):
    nyquist = sample_rate / 2.0
    b_src, a_src = butter(2, [80.0 / nyquist, 120.0 / nyquist], btype='band')
    src_bass = lfilter(b_src, a_src, signal)
    
    zero_crossings = np.where(np.diff(np.signbit(src_bass)))[0]
    sub_gen = np.copy(src_bass)
    
    flip = False
    for idx in zero_crossings:
        if flip:
            sub_gen[idx:] *= -1
        flip = not flip
        
    b_sub, a_sub = butter(2, 50.0 / nyquist, btype='low')
    forced_sub = lfilter(b_sub, a_sub, sub_gen)
    return signal + (forced_sub * 0.9)


def apply_ai_enhancer(signal, sample_rate):
    nyquist = sample_rate / 2.0
    b_hi, a_hi = butter(2, min(3000.0 / nyquist, 0.95), btype='high')
    highs = lfilter(b_hi, a_hi, signal)
    ai_harmonics = np.tanh(highs * 2.5) * 0.15
    
    envelope = np.abs(signal)
    b_env, a_env = butter(1, min(20.0 / nyquist, 0.95), btype='low')
    smooth_env = lfilter(b_env, a_env, envelope)
    return signal + ai_harmonics + (signal * smooth_env * 0.15)


def apply_eq(signal, sample_rate, eq_gains):
    freqs = [40, 80, 160, 400, 1000, 2500, 6000, 14000]
    out_signal = np.copy(signal)
    nyquist = sample_rate / 2.0
    
    for f, gain_db in zip(freqs, eq_gains):
        if abs(gain_db) < 0.1:
            continue
        gain_lin = 10.0 ** (gain_db / 20.0) - 1.0
        Q = 1.4
        w0 = f / nyquist
        if w0 >= 0.95:
            continue
        b, a = iirpeak(w0, Q)
        band = lfilter(b, a, signal)
        out_signal += band * gain_lin
    return out_signal


def patp_process_audio(input_file, output_file, bass_gain, space_gain, 
                        super_bass, forced_sub, sub_40hz_boost, philharmonic, ai_enhance, eq_gains):
    is_temp = False
    working_file = input_file
    try:
        working_file, is_temp = convert_to_wav_if_needed(input_file)

        sample_rate, data = wavfile.read(working_file)
        
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0

        if len(data.shape) < 2 or data.shape[1] != 2:
            return False, "Aplikacja obsługuje obecnie wyłącznie pliki STEREO (2 kanały)."

        L = data[:, 0]
        R = data[:, 1]

        if ai_enhance:
            L = apply_ai_enhancer(L, sample_rate)
            R = apply_ai_enhancer(R, sample_rate)

        if forced_sub:
            L = apply_sub_harmonic_synth(L, sample_rate, target_freq=40.0)
            R = apply_sub_harmonic_synth(R, sample_rate, target_freq=40.0)

        Mid = (L + R) * 0.5
        Side = (L - R) * 0.5

        Mid = apply_eq(Mid, sample_rate, eq_gains)

        if sub_40hz_boost > 0:
            nyquist = sample_rate / 2.0
            w0 = 40.0 / nyquist
            b_40, a_40 = iirpeak(w0, 1.2)
            sub_40_band = lfilter(b_40, a_40, Mid)
            Mid += sub_40_band * (sub_40hz_boost * 2.5)

        nyquist = sample_rate / 2.0
        cutoff = 70.0 if super_bass else 120.0
        b_low, a_low = butter(2, min(cutoff / nyquist, 0.95), btype='low')
        mid_lows = lfilter(b_low, a_low, Mid)
        
        if super_bass:
            bass_harmonics = np.tanh(mid_lows * 4.0) + (mid_lows * 1.2)
            effective_bass_gain = bass_gain * 1.8
        else:
            bass_harmonics = np.sin(mid_lows * np.pi * 0.5)
            effective_bass_gain = bass_gain * 0.5

        Mid_enhanced = Mid + (effective_bass_gain * bass_harmonics)

        shift_samples = int(0.002 * sample_rate)
        Side_shifted = np.roll(Side, shift_samples)
        Side_shifted[:shift_samples] = 0
        
        effective_space = space_gain * (1.4 if philharmonic else 1.0)
        Side_enhanced = Side + (effective_space * (Side - Side_shifted))

        if philharmonic:
            hall_delay = int(0.012 * sample_rate)
            hall_reverb = np.roll(Side, hall_delay)
            hall_reverb[:hall_delay] = 0
            b_hall, a_hall = butter(1, min(4000.0 / nyquist, 0.95), btype='low')
            hall_reverb = lfilter(b_hall, a_hall, hall_reverb)
            Side_enhanced += hall_reverb * 0.35

        L_out = Mid_enhanced + Side_enhanced
        R_out = Mid_enhanced - Side_enhanced

        output_data = np.vstack((L_out, R_out)).T
        
        max_peak = np.max(np.abs(output_data))
        if max_peak > 0.99:
            output_data = output_data / max_peak * 0.98

        output_data = np.clip(output_data, -1.0, 1.0)
        output_data_int16 = (output_data * 32767.0).astype(np.int16)
        
        wavfile.write(output_file, sample_rate, output_data_int16)
        return True, "Sukces"
    except Exception as e:
        return False, str(e)
    finally:
        if is_temp and os.path.exists(working_file):
            try:
                os.remove(working_file)
            except Exception:
                pass


class PATPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PATP Audio Processor v2.3 - Smart Overload Guard Edition")
        
        self.root.geometry("1280x750")
        self.root.minsize(960, 580)
        self.root.configure(bg="#121212")

        try:
            self.root.iconbitmap("logo.ico")
        except Exception:
            pass
        
        self.is_fullscreen = False
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        
        self.input_path = ""
        self.super_bass_var = tk.BooleanVar(value=False)
        self.forced_sub_var = tk.BooleanVar(value=True)
        self.philharmonic_var = tk.BooleanVar(value=False)
        self.ai_enhance_var = tk.BooleanVar(value=True)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background='#121212', foreground='#FFFFFF')
        self.style.configure('TButton', background='#1E1E1E', foreground='#FFFFFF', borderwidth=0, font=('Segoe UI', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#292929')])
        self.style.configure('Horizontal.TScale', background='#121212')
        self.style.configure('Vertical.TScale', background='#121212')
        self.style.configure('TCheckbutton', background='#121212', foreground='#00E5FF', font=('Segoe UI', 10, 'bold'))
        
        self.style.configure('TNotebook', background='#121212', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#1E1E1E', foreground='#AAAAAA', padding=[15, 8], font=('Segoe UI', 10, 'bold'))
        self.style.map('TNotebook.Tab', background=[('selected', '#00E5FF')], foreground=[('selected', '#000000')])

        main_container = tk.Frame(root, bg="#121212")
        main_container.pack(fill="both", expand=True, padx=40, pady=20)

        header = tk.Label(main_container, text="PATP AUDIO v2.3 SMART GUARD", bg="#121212", fg="#00E5FF", font=("Segoe UI", 24, "bold"))
        header.pack(pady=(0, 2))
        subheader = tk.Label(main_container, text="Psychoacoustic Audio Technologies Poland | Real-Time Clipping Guard & 3D Expansion Engine", bg="#121212", fg="#888888", font=("Segoe UI", 10, "italic"))
        subheader.pack(pady=(0, 10))

        file_frame = tk.Frame(main_container, bg="#121212")
        file_frame.pack(pady=10, fill="x")
        
        self.btn_browse = ttk.Button(file_frame, text="WYBIERZ PLIK (.WAV / .MP3 / .FLAC)", command=self.browse_file)
        self.btn_browse.pack(side="left", padx=(0, 15))
        
        self.lbl_file = tk.Label(file_frame, text="Nie wybrano pliku...", bg="#121212", fg="#AAAAAA", font=("Segoe UI", 11), anchor="w")
        self.lbl_file.pack(side="left", fill="x", expand=True)

        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True, pady=10)

        self.tab_main = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.tab_main, text=" BASS & STEREO 3D ")

        self.tab_eq = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.tab_eq, text=" KOREKTOR 8-PASM ")

        self.tab_ai = tk.Frame(self.notebook, bg="#121212")
        self.notebook.add(self.tab_ai, text=" 🤖 AI & STREAMING ")

        # --- TAB 1 (BASS & PRZESTRZEŃ STEREO) ---
        tk.Label(self.tab_main, text="🔊 Dedykowany Boost Pasma 40 Hz (Sub-Woofer):", bg="#121212", fg="#00E5FF", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        self.slider_40hz = ttk.Scale(self.tab_main, from_=0.0, to=1.0, value=0.5, orient="horizontal", command=lambda val: self.check_audio_overload())
        self.slider_40hz.pack(fill="x", padx=15, pady=2)

        self.chk_forced_sub = ttk.Checkbutton(self.tab_main, text="⚡ SYNTEZA SUB-BASU (WYMUSZENIE 40Hz ZE ŚRODKA BASU)", variable=self.forced_sub_var, command=self.check_audio_overload)
        self.chk_forced_sub.pack(anchor="w", padx=15, pady=(2, 10))

        tk.Label(self.tab_main, text="Moc ogólna TruBass:", bg="#121212", fg="#FFFFFF", font=("Segoe UI", 10)).pack(anchor="w", padx=15, pady=(2, 2))
        self.slider_bass = ttk.Scale(self.tab_main, from_=0.0, to=1.0, value=0.4, orient="horizontal", command=lambda val: self.check_audio_overload())
        self.slider_bass.pack(fill="x", padx=15, pady=2)

        self.chk_super_bass = ttk.Checkbutton(self.tab_main, text="🔥 SUPER MEGA TRUBASS (BOOST + HARMONICZNE)", variable=self.super_bass_var, command=self.check_audio_overload)
        self.chk_super_bass.pack(anchor="w", padx=15, pady=(2, 10))

        tk.Label(self.tab_main, text="🌐 Rozszerzenie Sceny Stereo (WOW Space 3D):", bg="#121212", fg="#00E5FF", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(10, 2))
        self.slider_space = ttk.Scale(self.tab_main, from_=0.0, to=1.0, value=0.5, orient="horizontal")
        self.slider_space.pack(fill="x", padx=15, pady=2)

        self.chk_philharmonic = ttk.Checkbutton(self.tab_main, text="🎻 TRYB FILHARMONIA (POGŁOS SALI KONCERTOWEJ)", variable=self.philharmonic_var)
        self.chk_philharmonic.pack(anchor="w", padx=15, pady=(2, 10))

        # --- TAB 2 (KOREKTOR 8-PASM) ---
        eq_header_frame = tk.Frame(self.tab_eq, bg="#121212")
        eq_header_frame.pack(fill="x", padx=15, pady=10)

        tk.Label(eq_header_frame, text="Korektor Graficzny (Start od 40 Hz)", bg="#121212", fg="#00E5FF", font=("Segoe UI", 11, "bold")).pack(side="left")
        
        btn_smile = tk.Button(eq_header_frame, text="😊 Uśmiech", bg="#1E1E1E", fg="#00E5FF", font=("Segoe UI", 9, "bold"), bd=1, relief="solid", command=self.preset_smile)
        btn_smile.pack(side="right", padx=5)

        btn_reset = tk.Button(eq_header_frame, text="Reset", bg="#1E1E1E", fg="#AAAAAA", font=("Segoe UI", 9), bd=1, relief="solid", command=self.preset_flat)
        btn_reset.pack(side="right")

        sliders_frame = tk.Frame(self.tab_eq, bg="#121212")
        sliders_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.eq_freqs = ["40Hz", "80Hz", "160Hz", "400Hz", "1kHz", "2.5kHz", "6kHz", "14kHz"]
        self.eq_sliders = []

        for i, freq in enumerate(self.eq_freqs):
            col_frame = tk.Frame(sliders_frame, bg="#121212")
            col_frame.pack(side="left", fill="both", expand=True)

            val_lbl = tk.Label(col_frame, text="0 dB", bg="#121212", fg="#AAAAAA", font=("Segoe UI", 9))
            val_lbl.pack(side="top")

            def make_cmd(l=val_lbl):
                return lambda val: (l.config(text=f"{float(val):+.1f}dB"), self.check_audio_overload())

            slider = ttk.Scale(col_frame, from_=12.0, to=-12.0, value=0.0, orient="vertical", command=make_cmd(val_lbl))
            slider.pack(side="top", fill="y", expand=True, pady=5)
            self.eq_sliders.append(slider)

            freq_lbl = tk.Label(col_frame, text=freq, bg="#121212", fg="#00E5FF", font=("Segoe UI", 9, "bold"))
            freq_lbl.pack(side="bottom")

        # --- TAB 3 (AI & STREAMING) ---
        ai_box = tk.LabelFrame(self.tab_ai, text=" AI Neural Enhancer ", bg="#121212", fg="#00E5FF", font=("Segoe UI", 11, "bold"), padx=15, pady=15)
        ai_box.pack(fill="x", padx=15, pady=10)

        self.chk_ai = ttk.Checkbutton(ai_box, text="⚡ Włącz rekonstrukcję spektralną AI w locie", variable=self.ai_enhance_var)
        self.chk_ai.pack(anchor="w")

        stream_box = tk.LabelFrame(self.tab_ai, text=" Pobieranie ze Streamingu (YouTube) ", bg="#121212", fg="#00E5FF", font=("Segoe UI", 11, "bold"), padx=15, pady=15)
        stream_box.pack(fill="x", padx=15, pady=10)

        tk.Label(stream_box, text="Wklej link YouTube:", bg="#121212", fg="#FFFFFF", font=("Segoe UI", 10)).pack(anchor="w")
        self.entry_url = tk.Entry(stream_box, bg="#1E1E1E", fg="#00E5FF", insertbackground="#FFFFFF", font=("Segoe UI", 10), bd=1, relief="solid")
        self.entry_url.pack(fill="x", pady=6)

        self.btn_download = tk.Button(stream_box, text="⬇️ Pobierz & Ustaw jako źródło", bg="#1E1E1E", fg="#00E5FF", font=("Segoe UI", 10, "bold"), command=self.download_stream)
        self.btn_download.pack(anchor="e", pady=5)

        # --- PASEK OSTRZEŻENIA PRZED PRZESTEREM ---
        self.lbl_warning = tk.Label(main_container, text="", bg="#121212", fg="#FFD700", font=("Segoe UI", 9, "bold"), wraplength=800)
        self.lbl_warning.pack(pady=(5, 2))

        # --- PRZYCISK PROCESUJ ---
        self.btn_process = tk.Button(main_container, text="PROCESUJ AUDIO", bg="#00E5FF", fg="#000000", font=("Segoe UI", 12, "bold"), activebackground="#00B2CC", bd=0, command=self.process_audio)
        self.btn_process.pack(fill="x", pady=(5, 0), ipady=8)

        # Inicjalne sprawdzenie suwaków
        self.check_audio_overload()

    def check_audio_overload(self, *args):
        """Dynamiczna analiza sumarycznej energii niski tonów w czasie rzeczywistym"""
        score = 0.0
        
        score += self.slider_40hz.get() * 2.5
        score += self.slider_bass.get() * 1.5
        
        if self.super_bass_var.get():
            score *= 1.4
        if self.forced_sub_var.get():
            score += 0.8
            
        eq_bass_boost = max(0, self.eq_sliders[0].get()) + max(0, self.eq_sliders[1].get())
        score += (eq_bass_boost / 6.0)

        if score > 3.2:
            self.lbl_warning.config(
                text="⚠️ OSTRZEŻENIE: Ustawiono bardzo wysoki poziom basu! Dźwięk może stracić czystość lub być przesterowany na mniejszych głośnikach."
            )
        else:
            self.lbl_warning.config(text="")

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)

    def exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)

    def preset_smile(self):
        smile_values = [8.0, 4.0, -1.0, -3.0, -1.5, 2.5, 4.5, 6.0]
        for slider, val in zip(self.eq_sliders, smile_values):
            slider.set(val)
        self.check_audio_overload()

    def preset_flat(self):
        for slider in self.eq_sliders:
            slider.set(0.0)
        self.check_audio_overload()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac")])
        if file_path:
            self.input_path = file_path
            self.lbl_file.config(text=os.path.basename(file_path))

    def download_stream(self):
        if not HAS_YTDLP:
            messagebox.showerror("Brak biblioteki yt-dlp", "Moduł pobierania z YouTube nie jest dostępny w tej kompilacji.")
            return

        url = self.entry_url.get().strip()
        if not url:
            messagebox.showwarning("Brak URL", "Wklej prawidłowy link z YouTube!")
            return

        self.btn_download.config(text="⏳ Pobieranie...", state="disabled")

        def run_dl():
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': 'stream_download.%(ext)s',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                def on_success():
                    self.input_path = "stream_download.wav"
                    self.lbl_file.config(text="stream_download.wav (Pobrany)")
                    self.btn_download.config(text="⬇️ Pobierz & Ustaw jako źródło", state="normal")
                    messagebox.showinfo("Sukces", "Pobrano i przekonwertowano audio z YouTube!")

                self.root.after(0, on_success)
            except Exception as e:
                def on_error():
                    self.btn_download.config(text="⬇️ Pobierz & Ustaw jako źródło", state="normal")
                    messagebox.showerror("Błąd pobierania", f"Nie udało się pobrać ścieżki.\n\nSzczegóły: {e}")

                self.root.after(0, on_error)

        threading.Thread(target=run_dl, daemon=True).start()

    def process_audio(self):
        if not self.input_path:
            messagebox.showwarning("Brak pliku", "Najpierw wybierz plik audio!")
            return
        
        default_out = os.path.splitext(os.path.basename(self.input_path))[0] + "_PATP_Processed.wav"
        output_path = filedialog.asksaveasfilename(initialfile=default_out, defaultextension=".wav", filetypes=[("Audio Files", "*.wav")])
        if not output_path:
            return

        self.btn_process.config(text="⏳ PRZETWARZANIE AUDIO (PROSZĘ CZEKAĆ)...", state="disabled")

        eq_gains = [s.get() for s in self.eq_sliders]

        def worker():
            success, message = patp_process_audio(
                self.input_path, 
                output_path, 
                self.slider_bass.get(), 
                self.slider_space.get(),
                self.super_bass_var.get(),
                self.forced_sub_var.get(),
                self.slider_40hz.get(),
                self.philharmonic_var.get(),
                self.ai_enhance_var.get(),
                eq_gains
            )

            def on_finish():
                self.btn_process.config(text="PROCESUJ AUDIO", state="normal")
                if success:
                    messagebox.showinfo("Sukces!", "Plik został pomyślnie przetworzony!")
                else:
                    messagebox.showerror("Błąd", message)

            self.root.after(0, on_finish)

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = PATPApp(root)
    root.mainloop()