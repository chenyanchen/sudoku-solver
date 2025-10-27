package cvservice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
)

type Client struct {
	url    *url.URL
	client *http.Client
}

func NewClient(_url string, client *http.Client) (*Client, error) {
	u, err := url.Parse(_url)
	if err != nil {
		return nil, fmt.Errorf("invalid url: %w", err)
	}

	if client == nil {
		client = http.DefaultClient
	}

	return &Client{url: u, client: client}, nil
}

func (c *Client) Recognize(ctx context.Context, req *RecognizeRequest) (*RecognizeResponse, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", req.File.Filename)
	if err != nil {
		return nil, fmt.Errorf("create form: %w", err)
	}

	file, err := req.File.Open()
	if err != nil {
		return nil, fmt.Errorf("open multipart file: %w", err)
	}
	defer file.Close()

	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("IO copying file: %w", err)
	}

	if err = writer.Close(); err != nil {
		return nil, fmt.Errorf("close multipart writer: %w", err)
	}

	_url := c.url.JoinPath("/recognize").String()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, _url, body)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	request.Header.Set("Content-Type", writer.FormDataContentType())

	response, err := c.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		resp, _ := io.ReadAll(response.Body)
		return nil, fmt.Errorf("server response status code: %d, body: %s", response.StatusCode, resp)
	}

	var resp RecognizeResponse
	if err = json.NewDecoder(response.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("decode response body: %w", err)
	}

	return &resp, nil
}
