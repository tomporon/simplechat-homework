# lambda/index.py
import json
import os
import boto3
import re  # 正規表現モジュールをインポート
from botocore.exceptions import ClientError
import urllib.request
import urllib.error 


# Lambda コンテキストからリージョンを抽出する関数
def extract_region_from_arn(arn):
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    if match:
        return match.group(1)
    return "us-east-1"  # デフォルト値

# グローバル変数としてクライアントを初期化（初期値）
bedrock_client = None

# モデルID
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

FASTAPI_BASE_URL = "https://b731-34-16-162-189.ngrok-free.app"
# FastAPIの推論エンドポイントパス
FASTAPI_GENERATE_PATH = "/generate" # 提示されたパス

def lambda_handler(event, context):
    # FastAPI_BASE_URLが設定されているか確認
    if not FASTAPI_BASE_URL:
        print("Error: FASTAPI_BASE_URL environment variable is not set.")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": "FastAPI base URL is not configured."
            })
        }

    # FastAPIのフルURLを構築
    fastapi_url = f"{FASTAPI_BASE_URL}{FASTAPI_GENERATE_PATH}"

    try:
        print("Received event:", json.dumps(event))

        # Cognitoで認証されたユーザー情報を取得（ログ出力のみ、API呼び出しには影響なし）
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

        # リクエストボディの解析
        body = json.loads(event.get('body', '{}')) # bodyが空の場合も考慮
        message = body.get('message')
        conversation_history = body.get('conversationHistory', [])

        if not message:
             print("Error: 'message' field is missing in the request body.")
             return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": False,
                    "error": "Message field is required in the request body."
                })
            }

        print("Processing message:", message)
        print(f"Calling FastAPI at: {fastapi_url}")

        # FastAPIに送るリクエストペイロードを構築
        # FastAPIの /generate エンドポイントが期待する形式に合わせる
        api_request_payload = {
            "prompt": message, # LambdaのmessageをFastAPIのpromptにマッピング
            "max_new_tokens": 512, # 提示されたスキーマの例からハードコード
            "do_sample": True,     # 提示されたスキーマの例からハードコード
            "temperature": 0.7,    # 提示されたスキーマの例からハードコード
            "top_p": 0.9         # 提示されたスキーマの例からハードコード
        }

        # リクエストボディをJSON形式にエンコード
        api_request_payload_bytes = json.dumps(api_request_payload).encode('utf-8')

        # urllib.request を使用してFastAPIにPOSTリクエストを送信
        try:
            # Requestオブジェクトを作成
            req = urllib.request.Request(
                url=fastapi_url,
                data=api_request_payload_bytes,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )

            # URLを開いてレスポンスを取得
            with urllib.request.urlopen(req) as response:
                # レスポンスボディを読み込み
                response_body_bytes = response.read()
                response_body_str = response_body_bytes.decode('utf-8')

            # FastAPIからのレスポンスをJSONとして解析
            api_response_json = json.loads(response_body_str)

            print("FastAPI response:", json.dumps(api_response_json, default=str))

            # FastAPIからの応答の検証と生成テキストの取得
            # FastAPIの応答形式 {"generated_text": "...", "response_time": 0} を想定
            assistant_response = api_response_json.get('generated_text')

            if assistant_response is None:
                 raise Exception(f"FastAPI response missing 'generated_text' key. Full response: {api_response_json}")

        except urllib.error.HTTPError as e:
            # HTTPエラーが発生した場合 (4xx, 5xxなど)
            error_message = f"HTTP Error calling FastAPI: {e.code} - {e.reason}"
            print(error_message)
            # FastAPIからのエラーレスポンスボディがあれば読み込む（validation errorなど）
            try:
                 error_response_body = e.read().decode('utf-8')
                 print(f"FastAPI Error Body: {error_response_body}")
                 # ここでエラーボディを解析して詳細をメッセージに追加することも可能
                 # 例: validation errorの場合は detail フィールドを見る
                 error_details = json.loads(error_response_body)
                 if 'detail' in error_details:
                      error_message += f" Details: {error_details['detail']}"
                 elif 'error' in error_details: # FastAPI側でカスタムエラーを返す場合など
                      error_message += f" Details: {error_details['error']}"

            except Exception: # エラーボディの読み込みや解析に失敗した場合
                 pass

            return {
                "statusCode": e.code, # FastAPIからのステータスコードを可能な限り返す
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": False,
                    "error": error_message
                })
            }
        except urllib.error.URLError as e:
            # URLエラー（ネットワーク到達不能、ホスト名解決失敗など）
            error_message = f"URL Error calling FastAPI: {e.reason}"
            print(error_message)
            return {
                "statusCode": 500,
                 "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": False,
                    "error": f"Failed to reach FastAPI: {error_message}"
                })
            }
        except json.JSONDecodeError:
             error_message = "Failed to decode JSON response from FastAPI"
             print(error_message)
             return {
                "statusCode": 500,
                 "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": False,
                    "error": error_message
                })
            }
        except Exception as e:
            # その他の予期せぬエラー
            error_message = f"An unexpected error occurred during FastAPI call or processing: {str(e)}"
            print("Error:", error_message)
            return {
                "statusCode": 500,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({
                    "success": False,
                    "error": error_message
                })
            }

        # Lambdaの応答用に会話履歴を更新
        # 元の履歴 + ユーザーメッセージ + アシスタントメッセージ (FastAPI応答)
        messages_for_response = conversation_history.copy()
        messages_for_response.append({"role": "user", "content": message})
        if assistant_response: # 生成テキストがあれば追加
             messages_for_response.append({"role": "assistant", "content": assistant_response})

        # 成功レスポンスの返却
        # Lambdaの応答形式は元の Bedrock 呼び出し時と同じ形式に保つ
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*", # クロスオリジン対応
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response, # FastAPIからの生成テキスト
                "conversationHistory": messages_for_response # 更新した会話履歴
            })
        }

    except Exception as error:
        # Lambdaハンドラレベルでのエラー（イベント解析失敗、FASTAPI_BASE_URLチェック前など）
        print("Error processing event or initial setup:", str(error))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"Internal Lambda Error: {str(error)}"
            })
        }
    # try:
    #     # コンテキストから実行リージョンを取得し、クライアントを初期化
    #     global bedrock_client
    #     if bedrock_client is None:
    #         region = extract_region_from_arn(context.invoked_function_arn)
    #         bedrock_client = boto3.client('bedrock-runtime', region_name=region)
    #         print(f"Initialized Bedrock client in region: {region}")
        
    #     print("Received event:", json.dumps(event))
        
    #     # Cognitoで認証されたユーザー情報を取得
    #     user_info = None
    #     if 'requestContext' in event and 'authorizer' in event['requestContext']:
    #         user_info = event['requestContext']['authorizer']['claims']
    #         print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
    #     # リクエストボディの解析
    #     body = json.loads(event['body'])
    #     message = body['message']
    #     conversation_history = body.get('conversationHistory', [])
        
    #     print("Processing message:", message)
    #     print("Using model:", MODEL_ID)
        
    #     # 会話履歴を使用
    #     messages = conversation_history.copy()
        
    #     # ユーザーメッセージを追加
    #     messages.append({
    #         "role": "user",
    #         "content": message
    #     })
        
    #     # Nova Liteモデル用のリクエストペイロードを構築
    #     # 会話履歴を含める
    #     bedrock_messages = []
    #     for msg in messages:
    #         if msg["role"] == "user":
    #             bedrock_messages.append({
    #                 "role": "user",
    #                 "content": [{"text": msg["content"]}]
    #             })
    #         elif msg["role"] == "assistant":
    #             bedrock_messages.append({
    #                 "role": "assistant", 
    #                 "content": [{"text": msg["content"]}]
    #             })
        
    #     # invoke_model用のリクエストペイロード
    #     request_payload = {
    #         "messages": bedrock_messages,
    #         "inferenceConfig": {
    #             "maxTokens": 512,
    #             "stopSequences": [],
    #             "temperature": 0.7,
    #             "topP": 0.9
    #         }
    #     }
        
    #     print("Calling Bedrock invoke_model API with payload:", json.dumps(request_payload))
        
    #     # invoke_model APIを呼び出し
    #     response = bedrock_client.invoke_model(
    #         modelId=MODEL_ID,
    #         body=json.dumps(request_payload),
    #         contentType="application/json"
    #     )
        
    #     # レスポンスを解析
    #     response_body = json.loads(response['body'].read())
    #     print("Bedrock response:", json.dumps(response_body, default=str))
        
    #     # 応答の検証
    #     if not response_body.get('output') or not response_body['output'].get('message') or not response_body['output']['message'].get('content'):
    #         raise Exception("No response content from the model")
        
    #     # アシスタントの応答を取得
    #     assistant_response = response_body['output']['message']['content'][0]['text']
        
    #     # アシスタントの応答を会話履歴に追加
    #     messages.append({
    #         "role": "assistant",
    #         "content": assistant_response
    #     })
        
    #     # 成功レスポンスの返却
    #     return {
    #         "statusCode": 200,
    #         "headers": {
    #             "Content-Type": "application/json",
    #             "Access-Control-Allow-Origin": "*",
    #             "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    #             "Access-Control-Allow-Methods": "OPTIONS,POST"
    #         },
    #         "body": json.dumps({
    #             "success": True,
    #             "response": assistant_response,
    #             "conversationHistory": messages
    #         })
    #     }
        
    # except Exception as error:
    #     print("Error:", str(error))
        
    #     return {
    #         "statusCode": 500,
    #         "headers": {
    #             "Content-Type": "application/json",
    #             "Access-Control-Allow-Origin": "*",
    #             "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    #             "Access-Control-Allow-Methods": "OPTIONS,POST"
    #         },
    #         "body": json.dumps({
    #             "success": False,
    #             "error": str(error)
    #         })
    #     }
