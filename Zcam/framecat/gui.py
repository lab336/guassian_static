import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from frame_core import extract_and_composite


class FrameCompareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("视频同帧对比工具")
        self.geometry("1400x1000")
        self.video_paths = []
        self.composite_img = None  # numpy array
        # 缩放与适应窗口变量
        self.var_fit_window = tk.BooleanVar(value=False)
        self.var_zoom = tk.DoubleVar(value=1.0)
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        btn_add = ttk.Button(top, text="添加视频", command=self.add_videos)
        btn_add.pack(side=tk.LEFT)
        btn_remove = ttk.Button(top, text="移除选中", command=self.remove_selected)
        btn_remove.pack(side=tk.LEFT, padx=4)
        btn_clear = ttk.Button(top, text="清空列表", command=self.clear_list)
        btn_clear.pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="帧号:").pack(side=tk.LEFT, padx=(20, 4))
        self.entry_frame = ttk.Entry(top, width=8)
        self.entry_frame.pack(side=tk.LEFT)
        ttk.Label(top, text="或 时间(秒):").pack(side=tk.LEFT, padx=(12, 4))
        self.entry_time = ttk.Entry(top, width=8)
        self.entry_time.pack(side=tk.LEFT)

        ttk.Label(top, text="高度:").pack(side=tk.LEFT, padx=(12, 4))
        self.entry_height = ttk.Entry(top, width=6)
        self.entry_height.insert(0, "360")
        self.entry_height.pack(side=tk.LEFT)

        ttk.Label(top, text="列数(0水平):").pack(side=tk.LEFT, padx=(12, 4))
        self.entry_cols = ttk.Entry(top, width=6)
        self.entry_cols.insert(0, "0")
        self.entry_cols.pack(side=tk.LEFT)

        self.var_allow_missing = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="越界用最后帧", variable=self.var_allow_missing).pack(side=tk.LEFT, padx=12)

        self.btn_run = ttk.Button(top, text="生成合成图", command=self.run_extract)
        self.btn_run.pack(side=tk.LEFT, padx=12)

        self.btn_save = ttk.Button(top, text="保存图片", command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)

        # 第二行控件框，避免拥挤
        top2 = ttk.Frame(self)
        top2.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(top2, text="缩放:").pack(side=tk.LEFT)
        zoom_scale = ttk.Scale(top2, from_=0.2, to=1.0, orient=tk.HORIZONTAL, variable=self.var_zoom, command=lambda v: self.on_zoom_change())
        zoom_scale.pack(side=tk.LEFT, padx=(4, 12))
        self.zoom_scale = zoom_scale
        ttk.Checkbutton(top2, text="适应窗口", variable=self.var_fit_window, command=self.on_fit_toggle).pack(side=tk.LEFT, padx=(0, 12))

        # 主体区域：可拖动窗格
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 列表区
        list_frame = ttk.Frame(main_pane, width=300) # 指定初始宽度
        main_pane.add(list_frame, weight=1)
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)

        # 图像显示区（可滚动）
        img_frame = ttk.LabelFrame(main_pane, text="合成预览")
        main_pane.add(img_frame, weight=4)
        self.canvas = tk.Canvas(img_frame, bg="#222")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw_image())
        self.img_scroll_x = ttk.Scrollbar(img_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.img_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.img_scroll_y = ttk.Scrollbar(img_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.img_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=self.img_scroll_x.set, yscrollcommand=self.img_scroll_y.set)

        # 状态栏
        self.status = tk.StringVar(value="就绪")
        # 底部状态与进度
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar = ttk.Label(status_frame, textvariable=self.status, anchor="w")
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=180)
        # 默认不显示进度条，按需 pack

    def add_videos(self):
        paths = filedialog.askopenfilenames(title="选择视频", filetypes=[("视频文件", "*.mp4;*.mov;*.mkv;*.avi;*.flv"), ("所有文件", "*.*")])
        if not paths:
            return
        for p in paths:
            if p not in self.video_paths:
                self.video_paths.append(p)
                self.listbox.insert(tk.END, os.path.basename(p))
        self.status.set(f"已添加 {len(paths)} 个，共 {len(self.video_paths)} 个视频")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for index in reversed(sel):
            # 从 listbox 获取的是文件名，需要找到完整路径来移除
            fname = self.listbox.get(index)
            path_to_remove = next((p for p in self.video_paths if os.path.basename(p) == fname), None)
            if path_to_remove:
                self.video_paths.remove(path_to_remove)
            self.listbox.delete(index)
        self.status.set(f"移除后共 {len(self.video_paths)} 个视频")

    def clear_list(self):
        self.video_paths.clear()
        self.listbox.delete(0, tk.END)
        self.status.set("列表已清空")

    def run_extract(self):
        if not self.video_paths:
            messagebox.showwarning("提示", "请先添加视频")
            return
        frame_text = self.entry_frame.get().strip()
        time_text = self.entry_time.get().strip()
        frame = None
        time_val = None
        if frame_text:
            try:
                frame = int(frame_text)
            except ValueError:
                messagebox.showerror("错误", "帧号需为整数")
                return
        if time_text:
            try:
                time_val = float(time_text)
            except ValueError:
                messagebox.showerror("错误", "时间需为数字")
                return
        if frame is None and time_val is None:
            messagebox.showerror("错误", "必须填写帧号或时间之一")
            return
        try:
            height = int(self.entry_height.get().strip())
            cols = int(self.entry_cols.get().strip())
        except ValueError:
            messagebox.showerror("错误", "高度/列数需为整数")
            return
        allow_missing = self.var_allow_missing.get()
        self.status.set("正在处理，请稍候...")
        self.btn_run.config(state=tk.DISABLED)
        # 显示并启动进度条
        if not self.progress.winfo_ismapped():
            self.progress.pack(side=tk.RIGHT, padx=4, pady=2)
        self.progress.start(10)
        threading.Thread(target=self._do_extract, args=(frame, time_val, height, cols, allow_missing), daemon=True).start()

    def _do_extract(self, frame, time_val, height, cols, allow_missing):
        try:
            composite = extract_and_composite(self.video_paths, frame=frame, time=time_val, height=height, cols=cols, allow_missing=allow_missing)
        except Exception as e:
            self.after(0, lambda: self._on_extract_error(e))
            return
        self.composite_img = composite
        self.after(0, self._on_extract_success)

    def _on_extract_error(self, err):
        messagebox.showerror("处理失败", str(err))
        self.status.set("失败: " + str(err))
        self.btn_run.config(state=tk.NORMAL)
        # 停止进度
        self.progress.stop()
        self.progress.pack_forget()

    def _on_extract_success(self):
        self.status.set(f"处理完成，尺寸 {self.composite_img.shape[1]}x{self.composite_img.shape[0]}")
        self.btn_run.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.progress.stop()
        self.progress.pack_forget()
        self.redraw_image()

    def redraw_image(self):
        if self.composite_img is None:
            return
        # 原始图像转为 PIL
        base_rgb = cv2.cvtColor(self.composite_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(base_rgb)

        # 根据适应窗口或缩放变量计算显示尺寸
        if self.var_fit_window.get():
            cw = max(self.canvas.winfo_width(), 1)
            ch = max(self.canvas.winfo_height(), 1)
            scale = min(cw / pil.width, ch / pil.height, 1.0)
        else:
            scale = float(self.var_zoom.get())

        disp_w = int(pil.width * scale)
        disp_h = int(pil.height * scale)
        if scale != 1.0:
            pil_disp = pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        else:
            pil_disp = pil

        self.tk_img = ImageTk.PhotoImage(pil_disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=(0, 0, pil_disp.width, pil_disp.height))

    def on_zoom_change(self):
        if self.var_fit_window.get():
            return  # 适应窗口时忽略手动缩放
        self.redraw_image()

    def on_fit_toggle(self):
        # 适应窗口切换时禁用或启用缩放滑条
        if self.var_fit_window.get():
            self.zoom_scale.state(["disabled"])
        else:
            self.zoom_scale.state(["!disabled"])
        self.redraw_image()

    def save_image(self):
        if self.composite_img is None:
            return
        out_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if not out_path:
            return
        ok = cv2.imwrite(out_path, self.composite_img)
        if ok:
            self.status.set(f"已保存: {out_path}")
        else:
            messagebox.showerror("错误", "保存失败")


def main():
    app = FrameCompareGUI()
    app.mainloop()


if __name__ == "__main__":
    main()