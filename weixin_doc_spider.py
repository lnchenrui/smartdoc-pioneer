import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
from log import logger
import random
from datetime import datetime

class WeixinDocSpider:
    def __init__(self):
        self.base_url = 'https://developer.work.weixin.qq.com'
        self.start_url = 'https://developer.work.weixin.qq.com/document/path/90664'
        self.output_dir = 'docs'
        self.visited_urls = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.total_urls = 0
        self.processed_urls = 0
        self.max_retries = 3
        self.delay_range = (1, 3)  # 随机延迟范围（秒）
        
        # 设置日志
        self.setup_logging()

    def setup_logging(self):
        logger.dir = 'logs'
        if not os.path.exists(logger.dir):
            os.makedirs(logger.dir)
        
        logger.file = os.path.join(logger.dir, f'spider_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logger.info,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logger.file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    async def fetch_page(self, session, url, retry_count=0):
        try:
            # 添加随机延迟
            delay = random.uniform(*self.delay_range)
            await asyncio.sleep(delay)
            
            async with session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logging.warning(f'Failed to fetch {url}: HTTP {response.status}')
                    
                    if retry_count < self.max_retries:
                        logger.info(f'Retrying {url} (attempt {retry_count + 1}/{self.max_retries})')
                        return await self.fetch_page(session, url, retry_count + 1)
                    return None
                    
        except Exception as e:
            logging.error(f'Error fetching {url}: {str(e)}')
            if retry_count < self.max_retries:
                logger.info(f'Retrying {url} (attempt {retry_count + 1}/{self.max_retries})')
                return await self.fetch_page(session, url, retry_count + 1)
            return None

    def parse_links(self, html, current_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/document/path/'):
                full_url = urljoin(self.base_url, href)
                links.add(full_url)
        return links

    def extract_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        
        # 检查是否存在错误提示图标
        error_icon = soup.find('div', class_='msg_iconArea')
        if error_icon:
            logger.info('Skipping page with error icon')
            return None
        
        # 找到主要内容区域
        main_content = soup.find(class_='ep-doc-area-cnt')
        if not main_content:
            return ''
            
        # 提取标题和更新时间
        title = main_content.find(class_='ep-doc-area-title')
        update_time = main_content.find(class_='ep-doc-area-subtitle')
        
        # 获取主要文档内容
        doc_content = main_content.find(class_='ep-doc-area-cherry')
        
        formatted_content = []
        
        # 添加标题和更新时间
        if title:
            formatted_content.append(f"# {title.get_text(strip=True)}\n")
        if update_time:
            formatted_content.append(f"{update_time.get_text(strip=True)}\n\n")
        
        if doc_content:
            # 处理各种内容元素
            for elem in doc_content.find_all(['h2', 'h3', 'h4', 'p', 'pre', 'table', 'ul', 'ol']):
                if elem.name.startswith('h'):
                    level = int(elem.name[1])
                    formatted_content.append(f"\n{'#' * level} {elem.get_text(strip=True)}\n")
                
                elif elem.name == 'p':
                    text = elem.get_text(strip=True)
                    if text:
                        formatted_content.append(f"\n{text}\n")
                
                elif elem.name == 'pre':
                    code = elem.get_text(strip=False)
                    formatted_content.append(f"\n```\n{code.strip()}\n```\n")
                
                elif elem.name == 'table':
                    rows = []
                    # 处理表头
                    headers = []
                    for th in elem.find_all(['th']):
                        headers.append(th.get_text(strip=True))
                    if headers:
                        rows.append(' | '.join(headers))
                        rows.append('-' * len(' | '.join(headers)))
                    
                    # 处理数据行
                    for tr in elem.find_all('tr'):
                        if not tr.find('th'):  # 跳过表头行
                            cols = [td.get_text(strip=True) for td in tr.find_all('td')]
                            if any(cols):
                                rows.append(' | '.join(cols))
                    
                    if rows:
                        formatted_content.append('\n' + '\n'.join(rows) + '\n')
                
                elif elem.name in ['ul', 'ol']:
                    formatted_content.append('\n')
                    for li in elem.find_all('li', recursive=False):
                        formatted_content.append(f"- {li.get_text(strip=True)}\n")
                    formatted_content.append('\n')
        
        content = ''.join(formatted_content)
        
        # 清理格式
        content = re.sub(r'\n{3,}', '\n\n', content)  # 规范化空行
        content = content.strip() + '\n'  # 确保文件以换行符结束
        
        return content

    def save_content(self, url, content):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 从URL中提取文件名
        filename = url.split('/')[-1] + '.txt'
        filepath = os.path.join(self.output_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            logger.info(f'File already exists, skipping: {filepath}')
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f'Source URL: {url}\n\n')
            f.write(content)

    async def crawl_page(self, session, url):
        if url in self.visited_urls:
            logger.info(f'URL already visited, skipping: {url}')
            return
        
        self.visited_urls.add(url)
        self.processed_urls += 1
        progress = f'[{self.processed_urls}/{self.total_urls if self.total_urls > 0 else "?"}]'
        logger.info(f'{progress} Crawling: {url}')
        
        html = await self.fetch_page(session, url)
        if html:
            try:
                # 提取并保存内容
                content = self.extract_content(html)
                if content is None:  # 检查是否应该跳过该页面
                    logger.info(f'Skipping page with error icon: {url}')
                    return
                    
                self.save_content(url, content)
                logger.info(f'{progress} Successfully saved content from {url}')
                
                # 获取新链接
                new_links = self.parse_links(html, url)
                if new_links:
                    self.total_urls = max(self.total_urls, len(self.visited_urls) + len(new_links))
                
                tasks = []
                for link in new_links:
                    if link not in self.visited_urls:
                        tasks.append(self.crawl_page(session, link))
                if tasks:
                    await asyncio.gather(*tasks)
            except Exception as e:
                logging.error(f'Error processing {url}: {str(e)}')
        else:
            logging.error(f'Failed to fetch content from {url}')

    async def run(self):
        async with aiohttp.ClientSession() as session:
            await self.crawl_page(session, self.start_url)

def main():
    spider = WeixinDocSpider()
    asyncio.run(spider.run())

if __name__ == '__main__':
    main()