import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx
import plotly.express as px
import pandas as pd
import os
import re
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import urllib.parse
import subprocess
from flask import request
import requests
import shutil
import json
import base64
import PyPDF2
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 OpenAI API 密钥
os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()

class DirectoryChangeHandler(FileSystemEventHandler):
    def __init__(self, trigger_update_func, base_dir):
        self.trigger_update_func = trigger_update_func
        self.base_dir = base_dir
        self.processed_files = set()
        self.update_directory_txt_files(self.base_dir)

    def update_directory_txt_files(self, base_path):
        for root, dirs, files in os.walk(base_path):
            dirs = [d for d in dirs if not d.startswith('.')]
            files = [f for f in files if not f.startswith('.')]
            if dirs:
                current_dir_name = os.path.basename(root.strip("/"))
                if current_dir_name:
                    txt_filename = f"{current_dir_name}.txt"
                    txt_path = os.path.join(root, txt_filename)
                    with open(txt_path, 'w', encoding='utf-8') as txt_file:
                        for d in sorted(dirs, key=natural_sort_key):
                            txt_file.write(d + "\n")

    def on_any_event(self, event):
        """
        当文件或文件夹发生任何变化时触发。
        """
        # 忽略 .DS_Store 文件的变化
        if event.src_path.endswith(".DS_Store"):
            return

        logging.info(f"检测到事件: {event.event_type} - {event.src_path}")

        if event.is_directory:
            self.trigger_update_func()
            super().on_any_event(event)
            self.update_index_file()  # 更新 index.txt 文件
        elif event.event_type in ['created', 'deleted', 'modified', 'moved']:
            self.trigger_update_func()

    def update_index_file(self):
        index_file_path = os.path.join(self.base_dir, "index.txt")
        directory_structure = self.generate_index_content(self.base_dir)
        with open(index_file_path, "w", encoding="utf-8") as index_file:
            index_file.write(directory_structure)
        logging.info(f"已更新 index.txt 文件: {index_file_path}")

    def generate_index_content(self, base_path, prefix=""):
        content = ""
        entries = sorted(os.listdir(base_path), key=natural_sort_key)
        for entry in entries:
            if entry == ".DS_Store":
                continue
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                content += f"{prefix}|-- {entry} (Folder)\n"
                content += self.generate_index_content(full_path, prefix + "|   ")
            else:
                content += f"{prefix}|-- {entry} (File)\n"
        return content

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        if not file_path.startswith(os.path.join(self.base_dir, "-1_初筛库")):
            return
        if file_path in self.processed_files:
            return
        self.processed_files.add(file_path)

        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                file_content = ""
                for page in pdf_reader.pages:
                    file_content += page.extract_text()

            file_content = file_content[:3000]
            print(file_content)
            # 分类目录
            target_folder = self.hierarchical_classification(file_content)
            if not target_folder:
                target_folder = "0_未分类"

            # 从全文中提取作者和标题
            first_author, article_title = self.extract_metadata_from_pdf(file_content)

            self.move_and_rename_file(
                file_path,
                self.base_dir,
                target_folder,
                first_author,
                article_title
            )
        except Exception as e:
            logging.error(f"处理文件时发生错误：{e}")

    def hierarchical_classification(self, pdf_text):
        current_path = self.base_dir
        while True:
            current_dir_name = os.path.basename(current_path.strip("/"))
            if not current_dir_name:
                current_dir_name = os.path.basename(current_path)

            txt_file_path = os.path.join(current_path, f"{current_dir_name}.txt")
            if not os.path.exists(txt_file_path):
                relative_final_folder = os.path.relpath(current_path, self.base_dir)
                return relative_final_folder if relative_final_folder != '.' else ''

            with open(txt_file_path, 'r', encoding='utf-8') as f:
                subfolders = [line.strip() for line in f if line.strip()]

            if not subfolders:
                relative_final_folder = os.path.relpath(current_path, self.base_dir)
                return relative_final_folder if relative_final_folder != '.' else ''

            chosen_folder = self.query_llm_for_subfolder(pdf_text, subfolders)
            if chosen_folder not in subfolders:
                return None
            current_path = os.path.join(current_path, chosen_folder)

    def query_llm_for_subfolder(self, pdf_text, subfolders):
        instruction = (
            "You are an expert in classifying academic papers into the most appropriate research category.\n"
            "I will provide you with a list of folder names (each folder name is a category). "
            "You must select exactly ONE folder name from the provided list that best matches the content of the paper.\n"
            "The paper content will be given below. Carefully read and understand the paper content.\n"
            "Then choose exactly one folder name from the 'Available folders' list that best fits the paper's subject.\n"
            "Output ONLY the folder name, without quotes or additional text.\n"
            "Please be reminded that ALL of the folders/folder names are AI-related."
            "If you are unsure, choose the closest matching folder from the list.\n"
        )

        folder_list_str = "\n".join(subfolders)
        prompt = (
            f"{instruction}\n\n"
            f"Available folders:\n{folder_list_str}\n\n"
            f"Paper content:\n{pdf_text}\n\n"
            f"Choose ONE folder name from the above list that best matches the paper's subject."
        )

        response = self.query_openai(prompt)
        print("LLM 分类结果:", response)
        return response

    def extract_metadata_from_pdf(self, pdf_text):
        """
        使用LLM从PDF文本中提取第一作者和文章标题。
        LLM需要输出有效的JSON格式，例如：
        {"first_author":"张三","article_title":"深度学习的研究进展"}
        如果无法解析，则使用默认值“未知作者”和“未知标题”。
        """
        prompt = (
            "You are an expert in reading academic papers. I will provide the full text of a paper. "
            "Your task is to extract the first author's name and the article title.\n"
            "Output MUST be a valid JSON object with keys \"first_author\" and \"article_title\".\n"
            "If the first author or the article title cannot be determined, use \"未知作者\" and \"未知标题\" respectively.\n"
            "No additional text, explanation, or formatting. Only output the JSON.\n\n"
            f"Paper content:\n{pdf_text}\n\n"
            "Extract and output JSON now."
        )

        response = self.query_openai(prompt)
        print('作者信息：', response)
        # 尝试解析JSON
        try:
            data = json.loads(response)
            first_author = data.get("first_author", "未知作者")
            article_title = data.get("article_title", "未知标题")
        except json.JSONDecodeError as e:
            logging.error(f"无法解析返回的JSON：{response}. 错误: {e}")
            first_author = "未知作者"
            article_title = "未知标题"

        return first_author, article_title

    def query_openai(self, prompt):
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in classifying academic papers and extracting metadata."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()

    def move_and_rename_file(self, file_path, base_dir, target_folder, first_author, article_title):
        target_path = os.path.join(base_dir, target_folder)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # 去除文件名中不允许的字符
        safe_author = re.sub(r'[\\/:*?"<>|]', '', first_author)
        safe_title = re.sub(r'[\\/:*?"<>|]', '', article_title)
        new_file_name = f"{safe_author}_{safe_title}.pdf"
        new_file_path = os.path.join(target_path, new_file_name)
        shutil.move(file_path, new_file_path)

    def categorize_top_level_folder(self, file_path):
        pass

    def categorize_second_level_folder(self, file_path):
        pass


class ApplePapersDashboard:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600&display=swap",
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
            ]
        )
        self.data_lock = threading.Lock()
        self.latest_data = self.generate_directory_structure(self.base_dir)
        self.update_flag = False
        self.observer = None

        self.app.layout = self.create_layout()
        self.register_callbacks()
        self.start_file_watcher()

        self.register_download_route()

    def register_download_route(self):
        @self.app.server.route('/open')
        def open_file():
            file_path = request.args.get('path', None)
            if not file_path:
                return "Missing file path.", 400

            abs_path = os.path.abspath(file_path)
            if not abs_path.startswith(os.path.abspath(self.base_dir)):
                return "Access denied.", 403
            if not os.path.exists(abs_path):
                return "File not found.", 404

            try:
                subprocess.Popen(["open", "-a", "Preview", abs_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                return f"Error opening file: {str(e)}", 500

            return "", 204

    def start_file_watcher(self):
        def trigger_update():
            with self.data_lock:
                self.latest_data = self.generate_directory_structure(self.base_dir)
                self.update_flag = True

        event_handler = DirectoryChangeHandler(
            trigger_update_func=trigger_update,
            base_dir=self.base_dir
        )
        self.observer = Observer()
        self.observer.schedule(event_handler, self.base_dir, recursive=True)
        self.observer.start()

    def generate_directory_structure(self, root_path):
        def scan_dir(path):
            try:
                entries = [e for e in os.scandir(path) if e.name != ".DS_Store" and not e.name.endswith(".txt")]
            except PermissionError:
                entries = []

            dirs = [e for e in entries if e.is_dir()]
            files = [e for e in entries if e.is_file()]

            dirs = sorted(dirs, key=lambda x: natural_sort_key(x.name))
            files = sorted(files, key=lambda x: natural_sort_key(x.name))

            items = []
            for d in dirs:
                children, count = scan_dir(d.path)
                items.append({
                    "name": d.name,
                    "type": "folder",
                    "children": children,
                    "papers_count": count
                })
            for f in files:
                items.append({"name": f.name, "type": "file"})
            total_papers = len(files) + sum(child.get('papers_count', 0) for child in items if child['type'] == 'folder')
            return items, total_papers

        children, total = scan_dir(root_path)
        return {
            "name": os.path.basename(root_path.rstrip('/')) or root_path.rstrip('/'),
            "type": "folder",
            "children": children,
            "papers_count": total
        }

    def parse_data(self, data, current_path):
        node = data
        for p in current_path:
            if 'children' in node:
                for c in node['children']:
                    if c['type'] == 'folder' and c['name'] == p:
                        node = c
                        break

        children = node.get('children', [])
        folders = [c for c in children if c['type'] == 'folder']
        files = [c for c in children if c['type'] == 'file']
        last_level = (len(folders) == 0)

        dir_rows = [{"name": f['name'], "papers_count": f.get('papers_count', 0)} for f in folders]
        file_rows = [{"filename": f['name']} for f in files]

        df_structure = pd.DataFrame(dir_rows) if dir_rows else pd.DataFrame(columns=['name', 'papers_count'])
        df_files = pd.DataFrame(file_rows) if file_rows else pd.DataFrame(columns=['filename'])
        return df_structure, df_files, last_level

    def create_layout(self):
        return dbc.Container([
            dcc.Location(id='url', refresh=False),

            html.Div([
                html.H1("AI Papers Archive", className="text-center apple-title"),
            ], className="mb-5"),

            dcc.Store(id='store-update-trigger', data=None),
            dcc.Store(id='store-data', data=self.latest_data),
            dcc.Store(id='store-current-path', data=[]),

            dcc.Interval(id='interval-component', interval=1000, n_intervals=0),

            dbc.Row([
                dbc.Col([
                    html.Nav(id='breadcrumb', className='breadcrumb', style={"font-family": "SF Pro Display, sans-serif", "font-size": "1rem"})
                ])
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("论文分布")),
                        dbc.CardBody([
                            dcc.Graph(id='papers-treemap', style={'height': '60vh'})
                        ])
                    ], style={"border-radius": "16px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("论文列表")),
                        dbc.CardBody([
                            dbc.Row(id='file-list', style={"gap": "10px"})
                        ])
                    ], style={"border-radius": "16px", "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)"})
                ], width=6)
            ])
        ], fluid=True, style={'padding': '2rem'})

    def register_callbacks(self):
        @self.app.callback(
            Output('store-update-trigger', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def check_update(n_intervals):
            with self.data_lock:
                if self.update_flag:
                    self.update_flag = False
                    return time.time()
            return dash.no_update

        @self.app.callback(
            Output('store-data', 'data'),
            Input('store-update-trigger', 'data')
        )
        def update_store_data(trigger):
            if trigger is not None:
                with self.data_lock:
                    return self.latest_data
            return dash.no_update

        @self.app.callback(
            Output('store-current-path', 'data'),
            Input('papers-treemap', 'clickData'),
            Input('url', 'pathname'),
            State('store-current-path', 'data')
        )
        def update_current_path(click_data, pathname, current_path):
            triggered_id = ctx.triggered_id
            if triggered_id == "papers-treemap" and click_data:
                if 'points' in click_data and len(click_data['points']) > 0:
                    point = click_data['points'][0]
                    if 'customdata' in point and point['customdata']:
                        clicked_label = point['customdata'][0]
                        return current_path + [clicked_label]
            elif triggered_id == "url":
                if pathname:
                    decoded_path = urllib.parse.unquote(pathname.strip("/"))
                    parts = decoded_path.split("/")
                    return parts if parts != [''] else []
            return current_path

        @self.app.callback(
            Output('papers-treemap', 'figure'),
            Output('file-list', 'children'),
            Input('store-data', 'data'),
            Input('store-current-path', 'data')
        )
        def update_horizontal_bar_and_file_list(data, current_path):
            if data is None:
                fig = px.bar(
                    x=[0],
                    y=["无数据"],
                    orientation="h",
                    title="论文数量分布",
                    labels={"x": "论文数量", "y": "文件夹名称"}
                )
                fig.update_traces(textfont_size=18)
                return fig, []

            df_structure, df_files, last_level = self.parse_data(data, current_path)
            if df_structure.empty:
                df_structure = pd.DataFrame([{
                    'name': '无子目录',
                    'papers_count': 0
                }])

            df_structure['original_name'] = df_structure['name']
            df_structure['display_name'] = df_structure['original_name'].apply(
                lambda x: re.sub(r'\(.*?\)', '', x).strip()
            )
            df_structure = df_structure.sort_values(by="papers_count", ascending=False)

            color_scale = [
                "#5E9EFF",
                "#87CEFA",
                "#4CD964",
                "#FF6B6B",
                "#FFD700"
            ]

            fig = px.bar(
                df_structure,
                x="papers_count",
                y="display_name",
                orientation="h",
                title="论文数量分布",
                color="papers_count",
                color_continuous_scale=color_scale,
                custom_data=['original_name'],
                labels={"papers_count": "论文数量", "display_name": "文件夹名称"}
            )
            fig.update_traces(
                texttemplate="%{x}",
                textposition="outside",
                marker=dict(
                    line=dict(width=0.5, color='black'),
                    opacity=0.8
                )
            )
            fig.update_layout(
                transition={
                    'duration': 800,
                    'easing': 'cubic-in-out'
                },
                plot_bgcolor='rgba(255,255,255,0.1)',
                paper_bgcolor='rgba(255,255,255,0)',
                title={
                    "text": "论文数量分布",
                    "font": {"size": 24, "color": "#333"},
                    "x": 0.5
                },
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="SF Pro Display, sans-serif"
                )
            )

            file_list_items = [
                html.Div(
                    html.A(
                        [
                            html.I(
                                className="fas fa-file-alt",
                                style={
                                    "font-size": "1.2em",
                                    "margin-right": "8px",
                                    "color": "#007aff"
                                }
                            ),
                            html.Span(
                                f,
                                style={
                                    "font-family": "SF Pro Display, sans-serif",
                                    "font-weight": "400",
                                    "font-size": "1rem",
                                    "color": "#333"
                                }
                            )
                        ],
                        href=f"/open?path={urllib.parse.quote(os.path.join(self.base_dir, *current_path, f))}",
                        target="_self",
                        style={
                            "text-decoration": "none",
                            "display": "flex",
                            "align-items": "center",
                            "padding": "8px 0",
                            "transition": "color 0.2s ease, background-color 0.2s ease",
                        }
                    ),
                    style={
                        "border-bottom": "1px solid #f0f0f0",
                        "padding": "5px 0",
                        "hover": {
                            "background-color": "#f7f7f7",
                            "cursor": "pointer"
                        }
                    }
                ) for f in df_files['filename']
            ]
            return fig, file_list_items

        @self.app.callback(
            Output('breadcrumb', 'children'),
            Input('store-current-path', 'data')
        )
        def update_breadcrumb(current_path):
            breadcrumb_items = [
                html.A("AI Papers Archive", href="/", className="breadcrumb-item", style={
                    "color": "#007aff",
                    "text-decoration": "none",
                    "font-weight": "500",
                    "transition": "color 0.3s ease"
                })
            ]
            for i, p in enumerate(current_path):
                breadcrumb_items.extend([
                    html.Span(" / ", className="breadcrumb-separator", style={
                        "margin": "0 8px",
                        "color": "#6e6e6e"
                    }),
                    html.A(p, href=f"/{urllib.parse.quote('/'.join(current_path[:i+1]))}",
                        className="breadcrumb-item",
                        style={
                            "color": "#007aff",
                            "text-decoration": "none",
                            "transition": "color 0.3s ease",
                            "hover": {
                                "color": "#5856d6"
                            }
                        }
                    )
                ])
            return breadcrumb_items

    def run(self, debug=False):
        self.app.run_server(debug=debug, host='0.0.0.0', port=8050)


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    parts = [int(p) if p.isdigit() else p for p in parts]
    return parts

def main():
    base_dir = "/Users/yp1017/AI-Papers-Archive"
    dashboard = ApplePapersDashboard(base_dir)
    dashboard.run(debug=True)

if __name__ == "__main__":
    main()