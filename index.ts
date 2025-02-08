#!/usr/bin/env node

// Core dependencies for MCP server and protocol handling
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListResourcesRequestSchema,
    ListToolsRequestSchema,
    ReadResourceRequestSchema,
    ListPromptsRequestSchema,
    GetPromptRequestSchema,
    Tool,
    Resource,
    McpError,
    ErrorCode,
    TextContent,
    ImageContent,
} from "@modelcontextprotocol/sdk/types.js";

// Web scraping and content processing dependencies
import { chromium, Browser, Page } from 'playwright';
import TurndownService from "turndown";
import type { Node } from "turndown";
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
// Add type declaration for window extensions
declare global {
    interface Window {
        chrome: {
            runtime: Record<string, unknown>;
            loadTimes: () => void;
            csi: () => void;
            app: Record<string, unknown>;
        };
        Notification: {
            permission: NotificationPermission;
            requestPermission?: (callback?: NotificationPermissionCallback) => Promise<NotificationPermission>;
        };
    }
}
// Initialize temp directory for screenshots
const SCREENSHOTS_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'mcp-screenshots-'));

// Initialize Turndown service for converting HTML to Markdown
// Configure with specific formatting preferences
const turndownService: TurndownService = new TurndownService({
    headingStyle: 'atx',       // Use # style headings
    hr: '---',                 // Horizontal rule style
    bulletListMarker: '-',     // List item marker
    codeBlockStyle: 'fenced',  // Use ``` for code blocks
    emDelimiter: '_',          // Italics style
    strongDelimiter: '**',     // Bold style
    linkStyle: 'inlined',      // Use inline links
});

// Custom Turndown rules for better content extraction
// Remove script and style tags completely
turndownService.addRule('removeScripts', {
    filter: ['script', 'style', 'noscript'],
    replacement: () => ''
});

// Preserve link elements with their href attributes
turndownService.addRule('preserveLinks', {
    filter: 'a',
    replacement: (content: string, node: Node) => {
        const element = node as HTMLAnchorElement;
        const href = element.getAttribute('href');
        return href ? `[${content}](${href})` : content;
    }
});

// Preserve image elements with their src and alt attributes
turndownService.addRule('preserveImages', {
    filter: 'img',
    replacement: (content: string, node: Node) => {
        const element = node as HTMLImageElement;
        const alt = element.getAttribute('alt') || '';
        const src = element.getAttribute('src');
        return src ? `![${alt}](${src})` : '';
    }
});

// Core interfaces for research data management
interface ResearchResult {
    url: string;             // URL of the researched page
    title: string;           // Page title
    content: string;         // Extracted content in markdown
    timestamp: string;       // When the result was captured
    screenshotPath?: string; // Path to screenshot file on disk
}

// Define structure for research session data
interface ResearchSession {
    query: string;              // Search query that initiated the session
    results: ResearchResult[];  // Collection of research results
    lastUpdated: string;        // Timestamp of last update
}

// Screenshot management functions
async function saveScreenshot(screenshot: string, title: string): Promise<string> {
    // Convert screenshot from base64 to buffer
    const buffer = Buffer.from(screenshot, 'base64');

    // Check size before saving
    const MAX_SIZE = 5 * 1024 * 1024;  // 5MB
    if (buffer.length > MAX_SIZE) {
        throw new McpError(
            ErrorCode.InvalidRequest,
            `Screenshot too large: ${Math.round(buffer.length / (1024 * 1024))}MB exceeds ${MAX_SIZE / (1024 * 1024)}MB limit`
        );
    }

    // Generate a safe filename
    const timestamp = new Date().getTime();
    const safeTitle = title.replace(/[^a-z0-9]/gi, '_').toLowerCase();
    const filename = `${safeTitle}-${timestamp}.png`;
    const filepath = path.join(SCREENSHOTS_DIR, filename);

    // Save the validated screenshot
    await fs.promises.writeFile(filepath, buffer);

    // Return the filepath to the saved screenshot
    return filepath;
}

// Fix the cleanupScreenshots function
async function cleanupScreenshots(): Promise<void> {
    try {
        const files = await fs.promises.readdir(SCREENSHOTS_DIR);
        await Promise.all(files.map((file: string) =>
            fs.promises.unlink(path.join(SCREENSHOTS_DIR, file))
        ));
        await fs.promises.rmdir(SCREENSHOTS_DIR);
    } catch (error) {
        console.error('Error cleaning up screenshots:', error);
    }
}

// Available tools for web research functionality
const TOOLS: Tool[] = [
    {
        name: "search_google",
        description: "Search Google for a query",
        inputSchema: {
            type: "object",
            properties: {
                query: { type: "string", description: "Search query" },
            },
            required: ["query"],
        },
    },
    {
        name: "visit_page",
        description: "Visit a webpage and extract its content",
        inputSchema: {
            type: "object",
            properties: {
                url: { type: "string", description: "URL to visit" },
                takeScreenshot: { type: "boolean", description: "Whether to take a screenshot" },
            },
            required: ["url"],
        },
    },
    {
        name: "take_screenshot",
        description: "Take a screenshot of the current page",
        inputSchema: {
            type: "object",
            properties: {},  // No parameters needed
        },
    },
];

// Define available prompt types for type safety
type PromptName = "agentic-research";

// Define structure for research prompt arguments
interface AgenticResearchArgs {
    topic: string;  // Research topic provided by user
}

// Configure available prompts with their specifications
const PROMPTS = {
    // Agentic research prompt configuration
    "agentic-research": {
        name: "agentic-research" as const,  // Type-safe name
        description: "Conduct iterative web research on a topic, exploring it thoroughly through multiple steps while maintaining a dialogue with the user",
        arguments: [
            {
                name: "topic",                                     // Topic argument specification
                description: "The topic or question to research",  // Description of the argument
                required: true                                     // Topic is mandatory
            }
        ]
    }
} as const;  // Make object immutable

// Global state management for browser and research session
let browser: Browser | undefined;                 // Puppeteer browser instance
let page: Page | undefined;                       // Current active page
let currentSession: ResearchSession | undefined;  // Current research session data

// Configuration constants for session management
const MAX_RESULTS_PER_SESSION = 100;  // Maximum number of results to store per session
const MAX_RETRIES = 3;                // Maximum retry attempts for operations
const RETRY_DELAY = 1000;             // Delay between retries in milliseconds

// Generic retry mechanism for handling transient failures
async function withRetry<T>(
    operation: () => Promise<T>,  // Operation to retry
    retries = MAX_RETRIES,        // Number of retry attempts
    delay = RETRY_DELAY           // Delay between retries
): Promise<T> {
    let lastError: Error;

    // Attempt operation up to max retries
    for (let i = 0; i < retries; i++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error as Error;
            if (i < retries - 1) {
                console.error(`Attempt ${i + 1} failed, retrying in ${delay}ms:`, error);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    throw lastError!;  // Throw last error if all retries failed
}

// Add a new research result to the current session with data management
function addResult(result: ResearchResult): void {
    // If no current session exists, initialize a new one
    if (!currentSession) {
        currentSession = {
            query: "Research Session",
            results: [],
            lastUpdated: new Date().toISOString(),
        };
    }

    // If the session has reached the maximum number of results, remove the oldest result
    if (currentSession.results.length >= MAX_RESULTS_PER_SESSION) {
        currentSession.results.shift();
    }

    // Add the new result to the session and update the last updated timestamp
    currentSession.results.push(result);
    currentSession.lastUpdated = new Date().toISOString();
}

// Safe page navigation with error handling and bot detection
async function safePageNavigation(page: Page, url: string): Promise<void> {
    try {
        // Step 1: Set cookies to bypass consent banner
        await page.context().addCookies([{
            name: 'CONSENT',
            value: 'YES+',
            domain: '.google.com',
            path: '/'
        }]);

        // Step 2: Initial navigation
        const response = await page.goto(url, {
            waitUntil: 'domcontentloaded',
            timeout: 15000
        });

        // Step 3: Basic response validation
        if (!response) {
            throw new Error('Navigation failed: no response received');
        }

        // Check HTTP status code; if 400 or higher, throw an error
        const status = response.status();
        if (status >= 400) {
            throw new Error(`HTTP ${status}: ${response.statusText()}`);
        }

        // Step 4: Wait for network to become idle or timeout
        await Promise.race([
            page.waitForLoadState('networkidle', { timeout: 5000 })
                .catch(() => {/* ignore timeout */ }),
            // Fallback timeout in case networkidle never occurs
            new Promise(resolve => setTimeout(resolve, 5000))
        ]);

        // Step 5: Security and content validation
        const validation = await page.evaluate(() => {
            const botProtectionExists = [
                '#challenge-running',     // Cloudflare
                '#cf-challenge-running',  // Cloudflare
                '#px-captcha',           // PerimeterX
                '#ddos-protection',       // Various
                '#waf-challenge-html'     // Various WAFs
            ].some(selector => document.querySelector(selector));

            // Check for suspicious page titles
            const suspiciousTitle = [
                'security check',
                'ddos protection',
                'please wait',
                'just a moment',
                'attention required'
            ].some(phrase => document.title.toLowerCase().includes(phrase));

            // Count words in the page content
            const bodyText = document.body.innerText || '';
            const words = bodyText.trim().split(/\s+/).length;

            // Return validation results
            return {
                wordCount: words,
                botProtection: botProtectionExists,
                suspiciousTitle,
                title: document.title
            };
        });

        // If bot protection is detected, throw an error
        if (validation.botProtection) {
            throw new Error('Bot protection detected');
        }

        // If the page title is suspicious, throw an error
        if (validation.suspiciousTitle) {
            throw new Error(`Suspicious page title detected: "${validation.title}"`);
        }

        // If the page contains insufficient content, throw an error
        if (validation.wordCount < 1) {
            throw new Error('Page contains insufficient content');
        }

    } catch (error) {
        // If an error occurs during navigation, throw an error with the URL and the error message
        throw new Error(`Navigation to ${url} failed: ${(error as Error).message}`);
    }
}

// Take and optimize a screenshot
async function takeScreenshotWithSizeLimit(page: Page): Promise<string> {
    const MAX_SIZE = 5 * 1024 * 1024;
    const MAX_DIMENSION = 1920;
    const MIN_DIMENSION = 800;

    // Set viewport size
    await page.setViewportSize({
        width: 1920,
        height: 1080
    });

    // Take initial screenshot
    let screenshot = await page.screenshot({
        type: 'png',
        fullPage: false
    });

    // Handle buffer conversion
    let buffer = screenshot;
    let attempts = 0;
    const MAX_ATTEMPTS = 3;

    // While screenshot is too large, reduce size
    while (buffer.length > MAX_SIZE && attempts < MAX_ATTEMPTS) {
        // Get current viewport size
        const viewport = page.viewportSize();
        if (!viewport) continue;

        // Calculate new dimensions
        const scaleFactor = Math.pow(0.75, attempts + 1);
        let newWidth = Math.round(viewport.width * scaleFactor);
        let newHeight = Math.round(viewport.height * scaleFactor);

        // Ensure dimensions are within bounds
        newWidth = Math.max(MIN_DIMENSION, Math.min(MAX_DIMENSION, newWidth));
        newHeight = Math.max(MIN_DIMENSION, Math.min(MAX_DIMENSION, newHeight));

        // Update viewport with new dimensions
        await page.setViewportSize({
            width: newWidth,
            height: newHeight
        });

        // Take new screenshot
        screenshot = await page.screenshot({
            type: 'png',
            fullPage: false
        });

        // Update buffer with new screenshot
        buffer = screenshot;

        // Increment retry attempts
        attempts++;
    }

    // Final attempt with minimum settings
    if (buffer.length > MAX_SIZE) {
        await page.setViewportSize({
            width: MIN_DIMENSION,
            height: MIN_DIMENSION
        });

        // Take final screenshot
        screenshot = await page.screenshot({
            type: 'png',
            fullPage: false
        });

        // Update buffer with final screenshot
        buffer = screenshot;

        // Throw error if final screenshot is still too large
        if (buffer.length > MAX_SIZE) {
            throw new McpError(
                ErrorCode.InvalidRequest,
                `Failed to reduce screenshot to under 5MB even with minimum settings`
            );
        }
    }

    // Convert Buffer to base64 string before returning
    return buffer.toString('base64');
}

// Initialize MCP server with basic configuration
const server: Server = new Server(
    {
        name: "webresearch",  // Server name identifier
        version: "0.1.6",     // Server version number
    },
    {
        capabilities: {
            tools: {},      // Available tool configurations
            resources: {},  // Resource handling capabilities
            prompts: {}     // Prompt processing capabilities
        },
    }
);

// Register handler for tool listing requests
server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS  // Return list of available research tools
}));

// Register handler for resource listing requests
server.setRequestHandler(ListResourcesRequestSchema, async () => {
    // Return empty list if no active session
    if (!currentSession) {
        return { resources: [] };
    }

    // Compile list of available resources
    const resources: Resource[] = [
        // Add session summary resource
        {
            uri: "research://current/summary",  // Resource identifier
            name: "Current Research Session Summary",
            description: "Summary of the current research session including queries and results",
            mimeType: "application/json"
        },
        // Add screenshot resources if available
        ...currentSession.results
            .map((r, i): Resource | undefined => r.screenshotPath ? {
                uri: `research://screenshots/${i}`,
                name: `Screenshot of ${r.title}`,
                description: `Screenshot taken from ${r.url}`,
                mimeType: "image/png"
            } : undefined)
            .filter((r): r is Resource => r !== undefined)
    ];

    // Return compiled list of resources
    return { resources };
});

// Register handler for resource content requests
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const uri = request.params.uri.toString();

    // Handle session summary requests for research data
    if (uri === "research://current/summary") {
        if (!currentSession) {
            throw new McpError(
                ErrorCode.InvalidRequest,
                "No active research session"
            );
        }

        // Return compiled list of resources
        return {
            contents: [{
                uri,
                mimeType: "application/json",
                text: JSON.stringify({
                    query: currentSession.query,
                    resultCount: currentSession.results.length,
                    lastUpdated: currentSession.lastUpdated,
                    results: currentSession.results.map(r => ({
                        title: r.title,
                        url: r.url,
                        timestamp: r.timestamp,
                        screenshotPath: r.screenshotPath
                    }))
                }, null, 2)
            }]
        };
    }

    // Handle screenshot requests
    if (uri.startsWith("research://screenshots/")) {
        const index = parseInt(uri.split("/").pop() || "", 10);

        // Verify session exists
        if (!currentSession) {
            throw new McpError(
                ErrorCode.InvalidRequest,
                "No active research session"
            );
        }

        // Verify index is within bounds
        if (isNaN(index) || index < 0 || index >= currentSession.results.length) {
            throw new McpError(
                ErrorCode.InvalidRequest,
                `Screenshot index out of bounds: ${index}`
            );
        }

        // Get result containing screenshot
        const result = currentSession.results[index];
        if (!result?.screenshotPath) {
            throw new McpError(
                ErrorCode.InvalidRequest,
                `No screenshot available at index: ${index}`
            );
        }

        try {
            // Read the binary data and convert to base64
            const screenshotData = await fs.promises.readFile(result.screenshotPath);

            // Convert Buffer to base64 string before returning
            const base64Data = screenshotData.toString('base64');

            // Return compiled list of resources
            return {
                contents: [{
                    uri,
                    mimeType: "image/png",
                    blob: base64Data
                }]
            };
        } catch (error: unknown) {
            // Handle error if screenshot cannot be read
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
            throw new McpError(
                ErrorCode.InternalError,
                `Failed to read screenshot: ${errorMessage}`
            );
        }
    }

    // Handle unknown resource types
    throw new McpError(
        ErrorCode.InvalidRequest,
        `Unknown resource: ${uri}`
    );
});

// Initialize MCP server connection using stdio transport
const transport = new StdioServerTransport();
server.connect(transport).catch((error) => {
    console.error("Failed to start server:", error);
    process.exit(1);
});

// Convert HTML content to clean, readable markdown format
async function extractContentAsMarkdown(
    page: Page,        // Puppeteer page to extract from
    selector?: string  // Optional CSS selector to target specific content
): Promise<string> {
    // Step 1: Execute content extraction in browser context
    const html = await page.evaluate((sel) => {
        // Handle case where specific selector is provided
        if (sel) {
            const element = document.querySelector(sel);
            // Return element content or empty string if not found
            return element ? element.outerHTML : '';
        }

        // Step 2: Try standard content containers first
        const contentSelectors = [
            'main',           // HTML5 semantic main content
            'article',        // HTML5 semantic article content
            '[role="main"]',  // ARIA main content role
            '#content',       // Common content ID
            '.content',       // Common content class
            '.main',          // Alternative main class
            '.post',          // Blog post content
            '.article',       // Article content container
        ];

        // Try each selector in priority order
        for (const contentSelector of contentSelectors) {
            const element = document.querySelector(contentSelector);
            if (element) {
                return element.outerHTML;  // Return first matching content
            }
        }

        // Step 3: Fallback to cleaning full body content
        const body = document.body;

        // Define elements to remove for cleaner content
        const elementsToRemove = [
            // Navigation elements
            'header',                    // Page header
            'footer',                    // Page footer
            'nav',                       // Navigation sections
            '[role="navigation"]',       // ARIA navigation elements

            // Sidebars and complementary content
            'aside',                     // Sidebar content
            '.sidebar',                  // Sidebar by class
            '[role="complementary"]',    // ARIA complementary content

            // Navigation-related elements
            '.nav',                      // Navigation classes
            '.menu',                     // Menu elements

            // Page structure elements
            '.header',                   // Header classes
            '.footer',                   // Footer classes

            // Advertising and notices
            '.advertisement',            // Advertisement containers
            '.ads',                      // Ad containers
            '.cookie-notice',            // Cookie consent notices
        ];

        // Remove each unwanted element from content
        elementsToRemove.forEach(sel => {
            body.querySelectorAll(sel).forEach(el => el.remove());
        });

        // Return cleaned body content
        return body.outerHTML;
    }, selector);

    // Step 4: Handle empty content case
    if (!html) {
        return '';
    }

    try {
        // Step 5: Convert HTML to Markdown
        const markdown = turndownService.turndown(html);

        // Step 6: Clean up and format markdown
        return markdown
            .replace(/\n{3,}/g, '\n\n')  // Replace excessive newlines with double
            .replace(/^- $/gm, '')       // Remove empty list items
            .replace(/^\s+$/gm, '')      // Remove whitespace-only lines
            .trim();                     // Remove leading/trailing whitespace

    } catch (error) {
        // Log conversion errors and return original HTML as fallback
        console.error('Error converting HTML to Markdown:', error);
        return html;
    }
}

// Validate URL format and ensure security constraints
function isValidUrl(urlString: string): boolean {
    try {
        // Attempt to parse URL string
        const url = new URL(urlString);

        // Only allow HTTP and HTTPS protocols for security
        return url.protocol === 'http:' || url.protocol === 'https:';
    } catch {
        // Return false for any invalid URL format
        return false;
    }
}

// Define result type for tool operations
type ToolResult = {
    content: (TextContent | ImageContent)[];  // Array of text or image content
    isError?: boolean;                        // Optional error flag
};

// Tool request handler for executing research operations
server.setRequestHandler(CallToolRequestSchema, async (request): Promise<ToolResult> => {
    // Initialize browser for tool operations
    const page = await ensureBrowser();

    switch (request.params.name) {
        // Handle Google search operations
        case "search_google": {
            // Extract search query from request parameters
            const { query } = request.params.arguments as { query: string };

            try {
                // Execute search with retry mechanism
                const results = await withRetry(async () => {
                    // Step 1: Navigate to Google search page
                    await safePageNavigation(page, 'https://www.google.com');

                    // Step 2: Find and interact with search input
                    await withRetry(async () => {
                        // Wait for any search input element to appear
                        await Promise.race([
                            // Try multiple possible selectors for search input
                            page.waitForSelector('input[name="q"]', { timeout: 5000 }),
                            page.waitForSelector('textarea[name="q"]', { timeout: 5000 }),
                            page.waitForSelector('input[type="text"]', { timeout: 5000 })
                        ]).catch(() => {
                            throw new Error('Search input not found - no matching selectors');
                        });

                        // Find the actual search input element
                        const searchInput = await page.$('input[name="q"]') ||
                            await page.$('textarea[name="q"]') ||
                            await page.$('input[type="text"]');

                        // Verify search input was found
                        if (!searchInput) {
                            throw new Error('Search input element not found after waiting');
                        }

                        // Step 3: Enter search query
                        await searchInput.click({ clickCount: 3 });  // Select all existing text
                        await searchInput.press('Backspace');        // Clear selected text
                        await searchInput.type(query);               // Type new query
                    }, 3, 2000);  // Allow 3 retries with 2s delay

                    // Step 4: Submit search and wait for results
                    await withRetry(async () => {
                        await Promise.all([
                            page.keyboard.press('Enter'),
                            page.waitForLoadState('networkidle', { timeout: 15000 }),
                        ]);
                    });

                    // Step 5: Extract search results
                    const searchResults = await withRetry(async () => {
                        const results = await page.evaluate(() => {
                            // Find all search result containers
                            const elements = document.querySelectorAll('div.g');
                            if (!elements || elements.length === 0) {
                                throw new Error('No search results found');
                            }

                            // Extract data from each result
                            return Array.from(elements).map((el) => {
                                // Find required elements within result container
                                const titleEl = el.querySelector('h3');            // Title element
                                const linkEl = el.querySelector('a');              // Link element
                                const snippetEl = el.querySelector('div.VwiC3b');  // Snippet element

                                // Skip results missing required elements
                                if (!titleEl || !linkEl || !snippetEl) {
                                    return null;
                                }

                                // Return structured result data
                                return {
                                    title: titleEl.textContent || '',        // Result title
                                    url: linkEl.getAttribute('href') || '',  // Result URL
                                    snippet: snippetEl.textContent || '',    // Result description
                                };
                            }).filter(result => result !== null);  // Remove invalid results
                        });

                        // Verify we found valid results
                        if (!results || results.length === 0) {
                            throw new Error('No valid search results found');
                        }

                        // Return compiled list of results
                        return results;
                    });

                    // Step 6: Store results in session
                    searchResults.forEach((result) => {
                        addResult({
                            url: result.url,
                            title: result.title,
                            content: result.snippet,
                            timestamp: new Date().toISOString(),
                        });
                    });

                    // Return compiled list of results
                    return searchResults;
                });

                // Step 7: Return formatted results
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(results, null, 2)  // Pretty-print JSON results
                    }]
                };
            } catch (error) {
                // Handle and format search errors
                return {
                    content: [{
                        type: "text",
                        text: `Failed to perform search: ${(error as Error).message}`
                    }],
                    isError: true
                };
            }
        }

        // Handle webpage visit and content extraction
        case "visit_page": {
            // Extract URL and screenshot flag from request
            const { url, takeScreenshot } = request.params.arguments as {
                url: string;                    // Target URL to visit
                takeScreenshot?: boolean;       // Optional screenshot flag
            };

            // Step 1: Validate URL format and security
            if (!isValidUrl(url)) {
                return {
                    content: [{
                        type: "text" as const,
                        text: `Invalid URL: ${url}. Only http and https protocols are supported.`
                    }],
                    isError: true
                };
            }

            try {
                // Step 2: Visit page and extract content with retry mechanism
                const result = await withRetry(async () => {
                    // Navigate to target URL safely
                    await safePageNavigation(page, url);
                    const title = await page.title();

                    // Step 3: Extract and process page content
                    const content = await withRetry(async () => {
                        // Convert page content to markdown
                        const extractedContent = await extractContentAsMarkdown(page);

                        // If no content is extracted, throw an error
                        if (!extractedContent) {
                            throw new Error('Failed to extract content');
                        }

                        // Return the extracted content
                        return extractedContent;
                    });

                    // Step 4: Create result object with page data
                    const pageResult: ResearchResult = {
                        url,      // Original URL
                        title,    // Page title
                        content,  // Markdown content
                        timestamp: new Date().toISOString(),  // Capture time
                    };

                    // Step 5: Take screenshot if requested
                    let screenshotUri: string | undefined;
                    if (takeScreenshot) {
                        // Capture and process screenshot
                        const screenshot = await takeScreenshotWithSizeLimit(page);
                        pageResult.screenshotPath = await saveScreenshot(screenshot, title);

                        // Get the index for the resource URI
                        const resultIndex = currentSession ? currentSession.results.length : 0;
                        screenshotUri = `research://screenshots/${resultIndex}`;

                        // Notify clients about new screenshot resource
                        server.notification({
                            method: "notifications/resources/list_changed"
                        });
                    }

                    // Step 6: Store result in session
                    addResult(pageResult);
                    return { pageResult, screenshotUri };
                });

                // Step 7: Return formatted result with screenshot URI if taken
                const response: ToolResult = {
                    content: [{
                        type: "text" as const,
                        text: JSON.stringify({
                            url: result.pageResult.url,
                            title: result.pageResult.title,
                            content: result.pageResult.content,
                            timestamp: result.pageResult.timestamp,
                            screenshot: result.screenshotUri ? `View screenshot via *MCP Resources* (Paperclip icon) @ URI: ${result.screenshotUri}` : undefined
                        }, null, 2)
                    }]
                };

                return response;
            } catch (error) {
                // Handle and format page visit errors
                return {
                    content: [{
                        type: "text" as const,
                        text: `Failed to visit page: ${(error as Error).message}`
                    }],
                    isError: true
                };
            }
        }

        // Handle standalone screenshot requests
        case "take_screenshot": {
            try {
                // Step 1: Capture screenshot with retry mechanism
                const screenshot = await withRetry(async () => {
                    // Take and optimize screenshot with default size limits
                    return await takeScreenshotWithSizeLimit(page);
                });

                // Step 2: Initialize session if needed
                if (!currentSession) {
                    currentSession = {
                        query: "Screenshot Session",            // Session identifier
                        results: [],                            // Empty results array
                        lastUpdated: new Date().toISOString(),  // Current timestamp
                    };
                }

                // Step 3: Get current page information
                const pageUrl = await page.url();      // Current page URL
                const pageTitle = await page.title();  // Current page title

                // Step 4: Save screenshot to disk
                const screenshotPath = await saveScreenshot(screenshot, pageTitle || 'untitled');

                // Step 5: Create and store screenshot result
                const resultIndex = currentSession ? currentSession.results.length : 0;
                addResult({
                    url: pageUrl,
                    title: pageTitle || "Untitled Page",  // Fallback title if none available
                    content: "Screenshot taken",          // Simple content description
                    timestamp: new Date().toISOString(),  // Capture time
                    screenshotPath                        // Path to screenshot file
                });

                // Step 6: Notify clients about new screenshot resource
                server.notification({
                    method: "notifications/resources/list_changed"
                });

                // Step 7: Return success message with resource URI
                const resourceUri = `research://screenshots/${resultIndex}`;
                return {
                    content: [{
                        type: "text" as const,
                        text: `Screenshot taken successfully. You can view it via *MCP Resources* (Paperclip icon) @ URI: ${resourceUri}`
                    }]
                };
            } catch (error) {
                // Handle and format screenshot errors
                return {
                    content: [{
                        type: "text" as const,
                        text: `Failed to take screenshot: ${(error as Error).message}`
                    }],
                    isError: true
                };
            }
        }

        // Handle unknown tool requests
        default:
            throw new McpError(
                ErrorCode.MethodNotFound,
                `Unknown tool: ${request.params.name}`
            );
    }
});

// Register handler for prompt listing requests
server.setRequestHandler(ListPromptsRequestSchema, async () => {
    // Return all available prompts
    return { prompts: Object.values(PROMPTS) };
});

// Register handler for prompt retrieval and execution
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
    // Extract and validate prompt name
    const promptName = request.params.name as PromptName;
    const prompt = PROMPTS[promptName];

    // Handle unknown prompt requests
    if (!prompt) {
        throw new McpError(ErrorCode.InvalidRequest, `Prompt not found: ${promptName}`);
    }

    // Handle agentic research prompt
    if (promptName === "agentic-research") {
        // Extract research topic from request arguments
        const args = request.params.arguments as AgenticResearchArgs | undefined;
        const topic = args?.topic || "";  // Use empty string if no topic provided

        // Return research assistant prompt with instructions
        return {
            messages: [
                // Initial assistant message establishing role
                {
                    role: "assistant",
                    content: {
                        type: "text",
                        text: "I am ready to help you with your research. I will conduct thorough web research, explore topics deeply, and maintain a dialogue with you throughout the process."
                    }
                },
                // Detailed research instructions for the user
                {
                    role: "user",
                    content: {
                        type: "text",
                        text: `I'd like to research this topic: <topic>${topic}</topic>

Please help me explore it deeply, like you're a thoughtful, highly-trained research assistant.

General instructions:
1. Start by proposing your research approach -- namely, formulate what initial query you will use to search the web. Propose a relatively broad search to understand the topic landscape. At the same time, make your queries optimized for returning high-quality results based on what you know about constructing Google search queries.
2. Next, get my input on whether you should proceed with that query or if you should refine it.
3. Once you have an approved query, perform the search.
4. Prioritize high quality, authoritative sources when they are available and relevant to the topic. Avoid low quality or spammy sources.
5. Retrieve information that is relevant to the topic at hand.
6. Iteratively refine your research direction based on what you find.
7. Keep me informed of what you find and let *me* guide the direction of the research interactively.
8. If you run into a dead end while researching, do a Google search for the topic and attempt to find a URL for a relevant page. Then, explore that page in depth.
9. Only conclude when my research goals are met.
10. **Always cite your sources**, providing URLs to the sources you used in a citation block at the end of your response.

You can use these tools:
- search_google: Search for information
- visit_page: Visit and extract content from web pages

Do *NOT* use the following tools:
- Anything related to knowledge graphs or memory, unless explicitly instructed to do so by the user.`
                    }
                }
            ]
        };
    }

    // Handle unsupported prompt types
    throw new McpError(ErrorCode.InvalidRequest, "Prompt implementation not found");
});

// In the ensureBrowser function, modify the initialization script:
async function ensureBrowser(): Promise<Page> {
    if (!browser) {
        browser = await chromium.launch({
            headless: true,
            args: [
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
                '--disable-setuid-sandbox',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-service-autorun',
                '--password-store=basic',
                '--system-developer-mode',
                '--enable-javascript',
                `--window-size=${1366 + Math.floor(Math.random() * 100)},${768 + Math.floor(Math.random() * 100)}`,
            ]
        });

        const context = await browser.newContext({
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport: { width: 1366, height: 768 },  // Set fixed viewport instead of null
            deviceScaleFactor: 1,
            javaScriptEnabled: true,
        });

        await context.addInitScript(() => {
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            Object.defineProperty(navigator, 'permissions', {
                get: () => ({
                    query: async () => ({ state: 'prompt' as PermissionState })
                })
            });

            window.chrome = {
                runtime: {},
                loadTimes: function(){},
                csi: function(){},
                app: {},
            };

            const originalToString = Error.prototype.toString;
            Error.prototype.toString = function(this: Error) {
                return originalToString.call(this).replace(/\n.*puppeteer.*\n/g, '\n');
            };

            if (!window.Notification) {
                const NotificationClass = function(title: string, options?: NotificationOptions) {
                    return {
                        title,
                        ...options
                    };
                } as unknown as typeof Notification;

                Object.defineProperty(NotificationClass, 'permission', {
                    value: 'default' as NotificationPermission,
                    writable: false,
                    configurable: false,
                    enumerable: true
                });

                NotificationClass.requestPermission = async () => 'default' as NotificationPermission;
                NotificationClass.prototype = {} as Notification;

                Object.defineProperty(window, 'Notification', {
                    value: NotificationClass,
                    writable: false,
                    configurable: false
                });
            }

            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        });

        page = await context.newPage();
        
        await page.route('**', async (route) => {
            const request = route.request();
            if (request.resourceType() === 'script') {
                route.continue({
                    headers: {
                        ...request.headers(),
                        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                        'sec-ch-ua-mobile': '?0',
                        'sec-ch-ua-platform': '"Windows"'
                    }
                });
            } else {
                route.continue();
            }
        });
    }

    if (!page) {
        const context = await browser.newContext();
        page = await context.newPage();
    }

    return page;
}


// Cleanup function
async function cleanup(): Promise<void> {
    try {
        // Clean up screenshots first
        await cleanupScreenshots();

        // Then close the browser
        if (browser) {
            await browser.close();
        }
    } catch (error) {
        console.error('Error during cleanup:', error);
    } finally {
        browser = undefined;
        page = undefined;
    }
}

// Register cleanup handlers
process.on('exit', cleanup);
process.on('SIGTERM', cleanup);
process.on('SIGINT', cleanup);
process.on('SIGHUP', cleanup);
