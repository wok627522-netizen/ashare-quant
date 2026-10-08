# 部署到 Streamlit Community Cloud（免费、24 小时在线）

这样部署后，**你的电脑关机也能访问**，网址形如 `https://xxx.streamlit.app`。

## 一、准备（约 10 分钟）

1. 注册 GitHub：<https://github.com/signup>（邮箱即可）
2. 回到本机，把这个 `cloud_deploy` 文件夹里的**全部内容**上传到一个新仓库：
   - 打开 <https://github.com/new>，仓库名填 `ashare-quant`，选 **Public**（Private 也行，Streamlit 支持授权），点 Create
   - 在新仓库页面点 **Add file → Upload files**
   - 把 `cloud_deploy` 里的 **app.py、requirements.txt、ashare_quant 文件夹、.streamlit 文件夹、.gitignore** 一起拖进去（注意：`.streamlit` 是隐藏文件夹，Windows 需先在资源管理器里勾选"显示隐藏项目"）
   - 下方 Commit changes 提交

## 二、部署到 Streamlit Cloud

1. 打开 <https://share.streamlit.io/>，用 **GitHub 账号登录**
2. 点 **Create app** → 选 **Deploy a public app from GitHub**
3. 填：
   - Repository：`你的用户名/ashare-quant`
   - Branch：`main`
   - Main file path：`app.py`
4. 点 **Advanced settings → Secrets**，把下面两行粘进去（把密码换成你自己的强密码）：
   ```toml
   PASSWORD = "你的访问密码"
   ASHARE_CLOUD = "1"
   ```
5. 点 **Deploy**，等 2~5 分钟（首次装依赖较慢）

部署完成后你会得到一个**永久网址**：`https://你的应用名.streamlit.app`

## 三、注意事项

- **密码**：写在 Secrets 里，不会出现在代码仓库中。本地运行仍用 `config/auth.json` 或环境变量。
- **云端默认股票池 = 主板活跃股前 300**（因为云端服务器在海外，访问新浪/东财较慢）。
  如需全主板，在页面左侧把「股票池范围」改成「沪深主板（全部）」——但首次加载可能要几分钟。
- **实盘通道（QMT / easytrader）在云端不可用**：它们需要你本机的券商客户端，云端只能看到「本地模拟盘 / 手动执行」。
- **数据不持久**：云端每次重启会清空 `data_cache`（行情缓存），属正常现象。
- 想省钱又要 24 小时：这个方案就是。

## 四、本地运行不受影响

原项目（本机）仍可双击 `一键启动公网.bat` 运行，功能完全一致。
